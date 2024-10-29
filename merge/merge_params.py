# -*- coding:utf-8 -*-

import torch
from qwen1_8.modeling_qwen import QWenLMHeadModel
from qwen1_8.tokenization_qwen import QWenTokenizer
import argparse
from peft import PeftModel

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device')
    parser.add_argument('--ori_model_dir', default="Qwen-1_8-chat/",type=str, help='model path')
    parser.add_argument('--model_dir', default="/root/auto-fs/DiaExtra/output_dir_qlora/epoch_1",type=str,help='qlora path')
    parser.add_argument('--save_model_dir', default="/root/auto-fs/DiaExtra/output_dir_qlora/epoch_1",
                        type=str, help='')
    return parser.parse_args()

def main():
    # 设置模型融合参数
    args = set_args()
    if args.device == "-1":
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    # 加载qwen原始模型
    base_model = QWenLMHeadModel.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16, device_map=device)
    tokenizer = QWenTokenizer.from_pretrained(args.ori_model_dir)
    # 加载Lora外挂参数
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    # 将外挂参数合并到原始参数中
    model = lora_model.merge_and_unload()
    # 将合并后的参数进行保存
    model.save_pretrained(args.save_model_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(args.save_model_dir)

if __name__ == "__main__":
    main()