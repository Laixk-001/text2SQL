import jsonlines
import glob
import pandas as pd
import os
import math
import multiprocessing as mp
import traceback
import tqdm
import itertools
import re
import collections
import argparse
from pathlib import Path
import json
import numpy as np
import itertools
import gc
import glob
import datasets
import ahocorasick
import subprocess
import hashlib
import random
import string
import nltk
import transformers

from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import torch
import torch.distributed
import logging
from transformers import set_seed
import bitsandbytes as bnb

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



class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args):
        return mp.get_logger().error(msg, *args) 

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can``
            # clean up
            raise

        # It was fine, give a normal answer
        return result

def truncate_prompt(prompt, max_num_tokens, tokenizer, side="right"):
    tokens = tokenizer.tokenize(prompt)
    num_tokens = len(tokens)
    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        prompt = tokenizer.convert_tokens_to_string(prompt_tokens)
        new_len = len(tokenizer.tokenize(prompt))
        if new_len > max_num_tokens:
            print(f'Number of tokens after truncation is greater than max tokens allowed: {new_len=} {num_tokens=}')
    return prompt
    
def read_file_from_position(args):
    filename, start_position, end_position, worker_id = args
    objs = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        current_position = find_next_line(f, start_position)
        f.seek(current_position)
        if current_position >= end_position:
            print(f"worker_id {worker_id} completed")
            return objs
        for cnt in tqdm.tqdm(itertools.count(), position=worker_id, desc=f"worker_id: {worker_id}"):
            line = f.readline()  
            if not line:
                break
            obj = json.loads(line)
            objs.append(obj)
            if f.tell() >= end_position:
                break
    print(f"worker_id {worker_id} completed")
    return objs



def filter_valid_code(code_list):
    def is_valid_python_code(code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    valid_codes = []
    for code in code_list:
        if is_valid_python_code(code):
            valid_codes.append(code)
    return valid_codes

def find_next_line(f, position):
    if position == 0:
        return position
    f.seek(position)
    f.readline()
    position = f.tell()
    return position

def multi_read(file_name = 'example.txt', workers = 32, chunk_size = None):    
    file_size = os.path.getsize(file_name)
    print(f"The size of {file_name} is: {file_size} bytes")
    if chunk_size:
        assert chunk_size > 0
        job_num = math.ceil(float(file_size) / chunk_size)
        positions = [chunk_size * i for i in range(job_num)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i) for i in range(job_num)]
        print(f"job num: {job_num}")
    else:
        chunk_size = math.ceil(float(file_size) / workers)
        positions = [chunk_size * i for i in range(workers)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i) for i in range(workers)]
    p = mp.Pool(workers)
    results = []
    for pos in start_positions:
        results.append(p.apply_async(MPLogExceptions(read_file_from_position), args=(pos,)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    print(f"Successfully Loading from {file_name}: {len(output_objs)} samples")
    return output_objs

def multi_read_fast(file_name = 'example.txt', workers = 32, chunk_size = None, task=read_file_from_position, args = []):    
    file_size = os.path.getsize(file_name)
    print(f"The size of {file_name} is: {file_size} bytes")
    if chunk_size:
        assert chunk_size > 0
        job_num = math.ceil(float(file_size) / chunk_size)
        positions = [chunk_size * i for i in range(job_num)]
        start_positions = [[file_name, positions[i], positions[i] + chunk_size, i] for i in range(job_num)]
        print(f"job num: {job_num}")
    else:
        chunk_size = math.ceil(float(file_size) / workers)
        positions = [chunk_size * i for i in range(workers)]
        start_positions = [[file_name, positions[i], positions[i] + chunk_size, i] for i in range(workers)]
    for pos in start_positions:
        pos.extend(args)
    p = mp.Pool(workers)
    results = []
    for pos in start_positions:
        results.append(p.apply_async(MPLogExceptions(task), args=(pos,)))
    p.close()
    p.join()
    print(f"Successfully Processing {file_name}")

def filter_code(text):
    def calculate_metrics(text):
        NON_ALPHA = re.compile("[^A-Za-z_0-9]")
        lines = text.strip().split('\n')
        line_lengths = [len(line) for line in lines]
        if len(lines) > 0:
            avg_line_length = sum(line_lengths) / len(lines)
            max_line_length = max(line_lengths)
        else:
            avg_line_length = 0
            max_line_length = 0
        alphanum_count = sum(c.isalnum() for c in text)
        alpha_count = sum(c.isalpha() for c in text)
        if len(text) > 0:
            alphanum_fraction = alphanum_count / len(text)
            alpha_fraction = alpha_count / len(text)
        else:
            alphanum_fraction = 0
            alpha_fraction = 0
        alpha_len = len(NON_ALPHA.split(text))
        char_len = len(text)
        tokens_num = len(text.split())
        return char_len, alpha_len, avg_line_length, max_line_length, alphanum_fraction, alpha_fraction, tokens_num
    char_len, alpha_len, avg_line_length, max_line_length, alphanum_fraction, alpha_fraction, tokens_num = calculate_metrics(text)
    if (1 < avg_line_length < 50) and (1 < max_line_length < 100) and (0.1 < alphanum_fraction < 1.0) and (0.1 < alpha_fraction < 1.0) and (10 < tokens_num < 1024):
        return False
    else:
        return True
    

def read_file_from_position_with_filter(args):
    filename, start_position, end_position, worker_id = args
    objs = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        current_position = find_next_line(f, start_position)
        f.seek(current_position)
        if current_position >= end_position:
            print(f"worker_id {worker_id} completed")
            return objs
        for cnt in tqdm.tqdm(itertools.count(), position=worker_id, desc=f"worker_id: {worker_id}"):
            line = f.readline()  
            if not line:
                break
            obj = json.loads(line)
            #if not filter_code(obj["text"]):
            objs.append(obj)
            if f.tell() >= end_position:
                break
    print(f"worker_id {worker_id} completed")
    return objs

def multi_read_with_filter(file_name = 'example.txt', workers = 32, chunk_size = None):    
    file_size = os.path.getsize(file_name)
    print(f"The size of {file_name} is: {file_size} bytes")
    if chunk_size:
        assert chunk_size > 0
        job_num = math.ceil(float(file_size) / chunk_size)
        positions = [chunk_size * i for i in range(job_num)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i) for i in range(job_num)]
        print(f"job num: {job_num}")
    else:
        chunk_size = math.ceil(float(file_size) / workers)
        positions = [chunk_size * i for i in range(workers)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i) for i in range(workers)]
    p = mp.Pool(workers)
    results = []
    for pos in start_positions:
        results.append(p.apply_async(MPLogExceptions(read_file_from_position_with_filter), args=(pos,)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    print(f"Successfully Loading from {file_name}: {len(output_objs)} samples")
    return output_objs

def read_jsonl_file(file_name, max_sentence=None):
    data = []
    with jsonlines.open(file_name, "r") as r:
        for i, obj in tqdm.tqdm(enumerate(r)):
            if max_sentence is not None and i >= max_sentence:
                return data
            data.append(obj)
    return data

def safe_read_jsonl_file(file_name, max_sentence=None):
    data = []
    with open(file_name, "r", encoding="utf-8", errors="ignore") as r:
        for i, line in tqdm.tqdm(enumerate(r)):
            try:
                obj = json.loads(line)
                if max_sentence is not None and i >= max_sentence:
                    return data
                data.append(obj)
            except:
                continue
    return data

def read_json_file(path):
    with open(path, "r") as r:
        objs = json.load(r)
    print(f"Successfully loading from {path}")
    return objs
    
def write_jsonl_file(objs, path, chunk_size = 1):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with jsonlines.open(path, "w", flush=True) as w:
        for i in tqdm.tqdm(range(0, len(objs), chunk_size)):
            w.write_all(objs[i: i + chunk_size])
    print(f"Successfully saving to {path}: {len(objs)}")


def read_jsonl_file(file_name, max_sentence=None):
    data = []
    with jsonlines.open(file_name, "r") as r:
        for i, obj in tqdm.tqdm(enumerate(r)):
            if max_sentence is not None and i >= max_sentence:
                return data
            data.append(obj)
    return data


def sentence_jaccard_similarity(sentence1, sentence2):
    def tokenize(sentence):
        """
        Tokenize the input sentence into a set of words.
        """
        # Convert to lowercase and split the sentence into words
        words = re.findall(r'\b\w+\b', sentence.lower())
        # Return the set of words
        return set(words)
    """
    Calculate the Jaccard Similarity between two sentences.
    """
    # Tokenize the sentences into sets of words
    set1 = tokenize(sentence1)
    set2 = tokenize(sentence2)
    
    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Compute Jaccard Similarity
    similarity = len(intersection) / len(union)
    return similarity

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def multi_tasks_from_file(file_name = 'example.txt', workers = 16, chunk_size = None, task = None, args = None):    
    file_size = os.path.getsize(file_name)
    print(f"The size of {file_name} is: {file_size} bytes")
    if chunk_size:
        assert chunk_size > 0
        job_num = math.ceil(float(file_size) / chunk_size)
        positions = [chunk_size * i for i in range(job_num)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i, args) for i in range(job_num)]
        print(f"job num: {job_num}")
    else:
        chunk_size = math.ceil(float(file_size) / workers)
        positions = [chunk_size * i for i in range(workers)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i, args) for i in range(workers)]
    p = mp.Pool(workers)
    results = []
    for pos in start_positions:
        results.append(p.apply_async(MPLogExceptions(task), args=(pos,)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    print(f"Successfully Loading from {file_name}: {len(output_objs)} samples")
    return output_objs

def multi_tasks_from_objs(objs, workers = 64, task=None, chunk_size=None, args=None):
    p = mp.Pool(workers)
    if chunk_size:
        results = []
        job_num = math.ceil(len(objs) / chunk_size)
        print(f"job num: {job_num}")
        for worker_id in range(job_num):
            results.append(p.apply_async(MPLogExceptions(task), args=(objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], worker_id, workers, args)))
    else:
        chunk_size = math.ceil(len(objs) / float(workers))
        results = []
        for worker_id in range(workers):
            results.append(p.apply_async(MPLogExceptions(task), args=(objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], worker_id, workers, args)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    return output_objs
    

def multi_write_jsonl_file(objs, path, workers = 16):
    chunk_size = math.ceil(len(objs) / workers)
    positions = [chunk_size * i for i in range(workers)]
    start_positions = [(objs[positions[i]: positions[i] + chunk_size], f"{path}-worker{i}.jsonl") for i in range(workers)]
    p = mp.Pool(workers)
    results = []
    for pos in start_positions:
        results.append(p.apply_async(MPLogExceptions(write_jsonl_file), args=(pos[0], pos[1])))
    p.close()
    p.join()
    p1 = subprocess.Popen(f"ls {path}-worker*.jsonl | sort -V | xargs cat > {path}", shell=True)
    p1.wait()
    print(f"Start merging to {path}")
    p2 = subprocess.Popen(f"rm {path}-worker*.jsonl", shell=True)
    print(f"Successfully Saving to {path}")


def extract_class_name(code):
    if re.search(r"public class\s+(\w*?)\s+{", code, flags=re.DOTALL) is not None:
        return re.search(r"class\s+(\w*?)\s+{", code, flags=re.DOTALL).group(1)
    else:
        return "Main"

class BM25:
    def __init__(self):
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

    def search(query = "text analysis in python"):  
        tokenized_query = word_tokenize(query.lower())
        doc_scores = bm25.get_scores(tokenized_query)
        best_docs = bm25.get_top_n(tokenized_query, corpus, n=3)
        return best_docs

def minihash_deduplicate(data):
    hash_set = set()
    deduped_data = []
    for item in tqdm.tqdm(data):
        hash_value = hashlib.md5(item["text"].encode()).hexdigest()
        if hash_value not in hash_set:
            deduped_data.append(item)
            hash_set.add(hash_value)
    return deduped_data

def contain_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def remove_comments(code, language = "python", remove_blank_line = True):
    if language == "python":
        code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        return code
    elif language == "java":
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*', '', code)
        return code
    elif language == "cpp":
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*', '', code, flags=re.DOTALL)
        # 匹配除了新行符之外的任何单个字符，现在匹配包括行结束符在内的任何单个字符
        # 匹配单行注释 //...
        # (?<!http:|https:) 避免删除URL中的双斜线
        #code = re.sub(r'(?<!http:|https:)\/\/.*', '', code)
    if remove_blank_line:
        code_lines = code.split("\n")
        code_lines = [c for c in code_lines if c != ""]
        code = "\n".join(code_lines)
    return code

