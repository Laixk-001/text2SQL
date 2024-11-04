# -*- coding:utf-8 -*-

import argparse
import json
from typing import Dict
import math
import transformers
from transformers import Trainer
import torch.distributed
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import torch.distributed as dist
from transformers import BitsAndBytesConfig
from utils.utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model,find_all_linear_names, evaluation, SupervisedDataset, DataCollatorForSupervisedDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import logging

# deepspeed.ops.op_builder.CPUAdamBuilder().load()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

def model_parse_args():
    model_parser = argparse.ArgumentParser()
    # 模型配置
    model_parser.add_argument("--model_name_or_path",type=str,default="/root/autodl-tmp/model/Qwen2_5_Coder_7B_Instruct",help="model name or path",required=True)
    return model_parser.parse_args()
def data_parse_args():
    data_parser = argparse.ArgumentParser()
    # 数据配置
    data_parser.add_argument("--train_path",default="/root/autodl-fs/DuSQL/text2sql_train_tokenizer_zh.json",type=str,help="")
    data_parser.add_argument("--test_path",default="/root/autodl-fs/DuSQL/text2sql_dev_tokenizer_zh.json",type=str,help="")
    data_parser.add_argument("--max_len",default=2048,type=int,help="")
    data_parser.add_argument("--model_max_length",default=1024,type=int,help="")
    data_parser.add_argument("--truncate_source",default=True,type=bool)
    data_parser.add_argument("--is_skip",action="store_true",help="")
    return data_parser.parse_args()
def train_parse_args():
    train_parser = argparse.ArgumentParser()
    # 训练配置
    train_parser.add_argument("--per_device_train_batch_size",default=16,type=int,help="")
    train_parser.add_argument("--per_device_eval_batch_size",default=4,type=int,help="")
    train_parser.add_argument("--learning_rate",default=1e-4,type=float,help="")
    train_parser.add_argument("--weight_decay",default=0.1,type=float,help="")
    train_parser.add_argument("--num_train_epochs",default=1,type=int,help="")
    train_parser.add_argument("--gradient_accumulation_steps",default=1,type=int,help="")
    train_parser.add_argument("--warmup_ratio",default=0.1,type=float,help="")
    train_parser.add_argument("--output_dir",default=None,type=str,help="")
    train_parser.add_argument("--seed", type=int, default=1234, help="")
    train_parser.add_argument("--local_rank", type=int, default=-1, help="")
    train_parser.add_argument("--show_loss_step", default=10, type=int, help="")
    train_parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    train_parser.add_argument("--save_model_step", default=None, type=int, help="")
    # DeepSpeed配置
    train_parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # QLoRA配置
    train_parser.add_argument("--lora_dim", type=int, default=8, help="")
    train_parser.add_argument("--lora_alpha", type=int, default=30, help="")
    train_parser.add_argument("--lora_dropout", type=float, default=0.1, help="")

    return train_parser.parse_args()

def is_master():
    return dist.get_rank() == 0

class LoggingCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_message = {
                "loss": logs.get("loss", None),
                "learning_rate": logs.get("learning_rate", None),
                "epoch": logs.get("epoch", None),
                "step": state.global_step
            }
            if is_master():
                print(log_message)

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_path, args=data_args)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.test_path, args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def train():
    # 设置模型训练参数
    data_args = data_parse_args()
    train_args = train_parse_args()
    model_args = model_parse_args()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        pad_token = '<|extra_0|>',
        eos_token = '<|im_end|>', #<|endoftext|>
        cache_dir = None,
        model_max_length = data_args.model_max_length,
        truncation = True,
        padding_side = "right",
        trust_remote_code = True
    )

    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=train_args, **data_module, callbacks=[LoggingCallback])
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=train_args.output_dir)
    
    
if __name__ == "__main__":
    train()