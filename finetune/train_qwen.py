# -*- coding:utf-8 -*-

import argparse
import json
import math
import transformers
import torch.distributed
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
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

def parse_args():
    parser = argparse.ArgumentParser()
    # 模型配置
    parser.add_argument("--model_name_or_path",type=str,default="/root/autodl-tmp/model/Qwen2_5_Coder_7B_Instruct",help="model name or path",required=True)
    # 数据配置
    parser.add_argument("--train_path",default="/root/autodl-tmp/data/text2sql_train_zh.json",type=str,help="")
    parser.add_argument("--test_path",default="/root/autodl-tmp/data/text2sql_dev_zh.json",type=str,help="")
    parser.add_argument("--max_len",default=2048,type=int,help="")
    parser.add_argument("--model_max_length",default=1024,type=int,help="")
    parser.add_argument("--truncate_source",default=True,type=bool)
    parser.add_argument("--is_skip",action="store_true",help="")
    # 训练配置
    parser.add_argument("--per_device_train_batch_size",default=16,type=int,help="")
    parser.add_argument("--learning_rate",default=1e-4,type=float,help="")
    parser.add_argument("--weight_decay",default=0.1,type=float,help="")
    parser.add_argument("--num_train_epochs",default=1,type=int,help="")
    parser.add_argument("--gradient_accumulation_steps",default=1,type=int,help="")
    parser.add_argument("--warmup_ratio",default=0.1,type=float,help="")
    parser.add_argument("--output_dir",default=None,type=str,help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--local_rank", type=str, default="0,1", help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # DeepSpeed配置
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # QLoRA配置
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def train():
    # 设置模型训练参数
    args = parse_args()
    # 判断是多卡训练还是单卡训练
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda",args.local_rank)
        deepspeed.init_distributed()
    # 获取当前进程的全局rank，用于区分主进程和辅助进程
    args.global_rank = torch.distributed.get_rank()
    # 如果当前进程是主进程，则初始化SummaryWriter，记录训练过程中的loss以及ppl
    if args.global_rank <= 0:
        tb_write = SummaryWriter()
    # 设置随机种子
    set_random_seed(args.seed)
    torch.distributed.barrier()
    # 加载qwen模型分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        pad_token = '<|extra_0|>',
        eos_token = '<|im_end|>', #<|endoftext|>
        cache_dir = None,
        model_max_length = args.model_max_length,
        truncation = True,
        padding_side = "right",
        trust_remote_code = True
    )
    # 加载qwen模型
    # device_map将模型映射到当前进程对应的GPU上
    device_map = {'': int(os.environ.get('LOCAL_RANK', '0'))}
    model_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    # 启用4-bit量化减少计算开销
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=model_config.torch_dtype,bnb_4bit_use_double_quant=False,bnb_4bit_quant_type="nf4",llm_int8_threshold=6.0,llm_int8_has_fp16_weight=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                            quantization_config=BitsAndBytesConfig(
                                                load_in_4bit=True,
                                                bnb_4bit_compute_dtype=model_config.torch_dtype,
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_quant_type="nf4",
                                                llm_int8_threshold=6.0,
                                                llm_int8_has_fp16_weight=False,
                                            ),
                                            torch_dtype=model_config.torch_dtype,
                                            device_map=device_map)
    # 以适应后续的低位量化训练，通常会冻结部分参数以减少计算开销
    model = prepare_model_for_kbit_training(model)
    # 找到模型中所有的全连接层
    lora_module_name = find_all_linear_names(model)
    # 设置Lora配置，并生成外挂可训练参数
    config = LoraConfig(r=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        target_modules=lora_module_name,
                        lora_dropout=args.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )
    model = get_peft_model(model,config)
    model.config.torch_dtype = torch.float32
    #打印可训练参数，确保只有指定模块参与训练
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print_rank_0(name, 0)
    print_trainable_parameters(model)

    # 加载模型训练所需要的数据，如果是多卡训练需要分布式加载数据
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.train_path, args = args)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.test_path, args = args)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
    
    # DataCollator使用分词器进行数据的批处理，将单个样本打包成模型可接受的批次格式
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                   collate_fn=data_collator,
                                   sampler=train_sampler,
                                   batch_size=args.per_device_train_batch_size)
    test_dataloader = DataLoader(test_dataset,
                                   collate_fn=data_collator,
                                   sampler=test_sampler,
                                   batch_size=args.per_device_train_batch_size)
    # 仅在主进程中输出，检查数据加载是否正常
    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)

    # 加载DeepSpeed配置文件，并修改
    with open(args.ds_file, "r", encoding="utf-8")as f:
        ds_config = json.load(f)
        # 每个GPU的训练微批大小
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        # 总训练批大小 = 微批 * GPU数量 * 梯度累积步数
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    # load optimizer
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
        # Adam的动量参数
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)
    ds_config["optimizer"]["params"]["eps"] = 1e-8
    ds_config["optimizer"]["params"]["weight_decay"] = 0.1    
    num_training_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print_rank_0("num_training_steps = {}".format(num_training_steps), args.global_rank)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    print_rank_0("num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)
    # 在预热阶段会逐步增加学习率，从而避免初始学习率过大导致的不稳定训练
    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1

    # 设置模型gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module,input,output):
                output.requires_grad_(True)
            
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # DeepSpeed对模型进行初始化
    model, optimizer, _ , lr_scheduler = deepspeed.initialize(model=model,args=args, config=ds_config, dist_init_required=True)
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 模型开始训练
    for epoch in range(args.num_train_epochs):
        print_rank_0("Beginning of Epoch {}/{}, Total Micro Batches {}".format(epoch + 1, args.num_train_epochs, len(train_dataloader)), args.global_rank)
        model.train()
        # 遍历所有数据
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            # 获取训练结果
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # 损失函数反向传播，计算每个参数的梯度
            model.backward(loss)
            tr_loss += loss.item()
            # 限制梯度的最大范数为1.0，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新模型参数
            model.step()
            # 当训练步数整除累积步数的时候，记录训练损失值和模型保存
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # 损失值记录
                if global_step % args.show_loss_step == 0:
                    if args.global_rank <= 0:
                        tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / (args.show_loss_step * args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss
                # 模型保存并验证测试集的PPL值
                if args.save_model_step is not None and global_step % args.save_model_step == 0:
                    ppl = evaluation(model, test_dataloader, device)
                    if args.global_rank <= 0:
                        tb_write.add_scalar("ppl", ppl, global_step)
                        print_rank_0("save_model_step-{}: ppl-{}".format(global_step, ppl), args.global_rank)
                    if args.global_rank <= 0:
                        save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                    model.train()
        # 每个epoch对模型进行一次测试， 记录测试集损失
        ppl = evaluation(model,test_dataloader,device)
        if args.global_rank <= 0:
            tb_write.add_scalar("ppl", ppl, global_step)
            print_rank_0("save_model_step-{}: ppl-{}".format(global_step, ppl), args.global_rank)
        if args.global_rank <= 0:
            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")

if __name__ == "__main__":
    train()