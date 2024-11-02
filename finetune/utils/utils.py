import jsonlines
import pandas as pd
import os
import multiprocessing as mp
import tqdm
from pathlib import Path
import numpy as np
import random
import transformers

from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import torch
import torch.distributed
import logging
from transformers import set_seed
import bitsandbytes as bnb

IGNORE_INDEX = -1

def read_jsonl_file(file_name, max_sentence=None):
    data = []
    with jsonlines.open(file_name, "r") as r:
        for i, obj in tqdm.tqdm(enumerate(r)):
            if max_sentence is not None and i >= max_sentence:
                return data
            data.append(obj)
    return data

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if data_path.endswith(".npy"):
            list_data_dict = np.load(data_path, allow_pickle=True)
        else:
            list_data_dict = read_jsonl_file(data_path)
        logging.info("Loading tokenized sentences...")
        def truncate(sentence):
            return torch.tensor(sentence[:args.model_max_length] + [tokenizer.eos_token_id] if len(sentence) > args.model_max_length else sentence)
        if args.truncate_source:
            self.input_ids = [truncate(example["input_ids"]) for example in list_data_dict]
            self.labels = [truncate(example["label"]) for example in list_data_dict]
        else:
            self.input_ids = [torch.tensor(example["input_ids"]) for example in list_data_dict if len(example["input_ids"]) < args.model_max_length]
            self.labels = [torch.tensor(example["label"]) for example in list_data_dict if len(example["input_ids"]) < args.model_max_length]
        print(f"Samples: {len(list_data_dict)} -> {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:        
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def print_trainable_parameters(model):
    """打印可训练的参数"""
    trainable_parameters = 0
    all_param = 0
    # 返回生成器包含参数名称和参数值的元组
    for _, param in model.named_parameters():
        # 计算该参数包含的元素数量（张量大小）
        num_params = param.numel()
        # 检查该参数对象是否具有ds_numel属性
        if num_params == 0 and hasattr(param, "ds_numel"):
            # 为了处理某些参数分布式存储的情况，ds_numel为该参数的实际大小
            num_params = param.ds_numel
        
        all_param += num_params
        # 如果该参数是可以训练的
        if param.requires_grad:
            trainable_parameters += num_params
    print("trainable params:{} || all params: {} || trainable%: {}".format(trainable_parameters, all_param, 100 * trainable_parameters / all_param))

def print_rank_0(msg, rank=0):
    """多卡训练时，打印rank=0上的信息"""
    if rank <= 0:
        return (msg)

def to_device(batch, device):
    """将Batch内的数据内容，设置到对应的device上"""
    output = {}
    for k,v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_model(model, tokenizer, output_dir, model_name):
    """模型保存，保存模型和对应的分词器"""
    save_dir = os.path.join(output_dir, model_name)
    model.save_pretrained(save_dir, torch_dtype=torch.float16)
    tokenizer.save_pretrained(save_dir)

def find_all_linear_names(model):
    """找到模型中所有的线性层"""
    cls = bnb.nn.Linear4bit
    # 用于存储符合条件的线性层名称
    lora_module_names = set()
    # 遍历模型所有的模块，返回模块名称及其名称的生成器
    for name, module in model.named_modules():
        # 判断模块是否属于cls对应的类
        if isinstance(module,cls):
            # 会分割成诸如：['layer1','linear']
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # lm_head为模型的输出层，不是普通的线性层，要排除
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def evaluation(model, eval_dataloader, device):
    """模型验证，计算验证集的PPL值"""
    # 调整为评估模式，会禁用dropout等训练时特定的行为
    model.eval()
    total_loss = 0
    # 获得当前的批次和数据，tqdm的单位为batch，总共有len(eval_dataloader)次
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader),unit="batch"):
        batch = to_device(batch, device)
        # 不需要计算梯度，节省内存并加速
        with torch.no_grad():
            # 将batch的参数传递给模型的前向函数
            outputs = model(**batch, use_cache=False)
            loss=outputs.loss
            total_loss += loss.float()
    total_loss = total_loss /(step + 1)

    try:
        perplexity = torch.exp(total_loss)
    except OverflowError:
        perplexity = float("inf")
    
    # 对PPL进行分布式的平均操作，并转换为python的标量
    try:
        perplexity = get_all_reduce_mean(perplexity).item()
    except:
        pass
    model.train()
    return perplexity

def get_all_reduce_mean(tensor):
    """在分布式训练中计算tensor的全局平均值"""
    # 分布式操作，将张量在所有设备上求和
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # 求和后的张量厨艺进程的总数，得到全局的平均值
    tensor = tensor / torch.distributed.get_world_size()
    return tensor



