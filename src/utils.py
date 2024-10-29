# -*- coding:utf-8 -*-

import torch
import random
import numpy as np
import torch.distributed
from transformers import set_seed
import json
import os
from torch.utils.data import Dataset
import bitsandbytes as bnb
from tqdm import tqdm
import math

class QwenPromptDataSet(Dataset):
    """对话要素抽取所需的数据类"""

    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        """
        初始化函数
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_len: 模型训练最大长度
            max_src_len: input的最大长度
            is_skip: 不符合长度标准的数据是否跳过
        """
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.max_src_len = int(max_src_len)
        self.is_skip = is_skip
        self.nl_tokens = self.tokenizer.encode("\n")
        self.all_data = self.load_data(data_path)

    def load_data(self, data_path):
        """
        加载原始数据，生成满足模型需求的数据，并剔除不达标的数据
        Args:
            data_path: 原始数据路径
        
        """
        self.all_data = []
        skip_data_number = 0
        
        # 遍历文件中的每一个样本
        with open(data_path, "r", encoding="utf-8")as f:
            for i, line in enumerate(f):
                sample = json.loads(line.strip())
                # 通过convert_feature函数将每一条数据进行索引化，生成模型所需要的input_ids和labels
                input_ids, labels, skip_flag = self.convert_feature(sample)
                # 跳过不符合标准的数据
                if self.is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids":input_ids, "labels":labels})
        print("the number of skipping data is {}, the proportion is {}".format(skip_data_number, skip_data_number / (
                len(self.all_data) + skip_data_number)))
        return self.all_data
    def _tokenize_str(self, role, content):
        """
        返回角色+内容文本和对应tokenizer的拼接
        Args:
            role: 角色
            content: 内容

        """
        role_content = f"{role}\n{content}"
        tokenizer_of_role_content = self.tokenizer.encode(role,allow_special=set()) + self.nl_tokens + self.tokenizer.encode(content, allow_special=set())
        return role_content, tokenizer_of_role_content
    
    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 包含提示词、输入内容、输出内容的字典， 格式为{"instruction":instruction, "input":input, "output":output}

        """
        skip_flag = False
        im_strat_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        # 构造qwen模型所需要的系统指令内容
        sys_prompt = "You are a helpful assistant."
        system_text, system_tokens_part = self._tokenize_str("system", sys_prompt)
        system_tokens = im_strat_tokens + system_tokens_part + im_end_tokens

        input_ids = []
        labels = []
        # 构建用户输入内容
        prompt_ids = im_strat_tokens + self._tokenize_str("user", sample["instruction"]+ sample["input"])[1] + im_end_tokens
        # 当用户输入内容长度超过最大长度时，进行向前截断，并生成对应的label
        if len(prompt_ids) > self.max_src_len:
            input_ids = self.nl_tokens + prompt_ids[:self.max_src_len - 1] + [prompt_ids[-1]]
            labels = [-100] * (len(input_ids))
            skip_flag = True
        else:
            input_ids.extend(self.nl_tokens + prompt_ids)
            labels.extend([-100] * (len(prompt_ids) + len(self.nl_tokens)))
        assert len(input_ids) == len(labels)
        # 构建模型输出内容
        output_id = im_strat_tokens + self._tokenize_str("assistant", sample["output"])[1] + im_end_tokens
        # 当模型输出内容长度超过最大长度时， 向前截断
        # print("input_ids:",len(input_ids))
        # print("system_tokens:",len(system_tokens))
        # print("max_len:",self.max_len)
        max_tgt_len = self.max_len - len(input_ids) - len(system_tokens)
        if len(output_id) > max_tgt_len:
            output_id = output_id[:max_tgt_len - 1] + [output_id[-1]]
            skip_flag = True
        # 将系统指令、 用户输入、 模型输出进行拼接， 构建完整的模型训练所需数据
        input_ids = system_tokens + input_ids + self.nl_tokens + output_id
        labels = [-100] * len(system_tokens) + labels + [-100] * (1 + len(self.nl_tokens)) + output_id[1:]
        
        assert len(input_ids) == len(labels)
        assert len(input_ids) <= self.max_len

        return input_ids, labels, skip_flag

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,item):
        instance = self.all_data[item]
        return instance

class DataCollator(object):
    """DataLoader所需DataCollator类"""
    def __init__(self, tokenizer):
        """
        初始化函数
        Args:
            tokenizer: 分词器
        """
        self.tokenizer = tokenizer
        # padding标记
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, batch):
        """
        回调函数
        Args:
            batch: 输入数据
        
        """
        # 获取一个Batch内的最大长度
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = math.ceil(max(lengths) / 8)* 8
        # 遍历Batch内容
        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]
            # 将Batch内的数据填充到最大长度
            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
        #将结果转化为torch.tensor并返回
        return {"input_ids":torch.tensor(input_ids_batch,dtype=torch.long),"labels":torch.tensor(labels_batch,dtype=torch.long)}

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