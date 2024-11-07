
## LoRA训练
- 单机两卡训练
```shell
export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 5545 train.py --train_path /root/autodl-fs/DuSQL/text2sql_train_tokenizer_zh.json  \
                                                                   --test_path /root/autodl-fs/DuSQL/text2sql_dev_tokenizer_zh.json  \
                                                                   --model_name_or_path /root/autodl-fs/Qwen2_5_Coder_7B_Instruct/  \
                                                                   --per_device_train_batch_size 2  \
                                                                   --max_len 2048  \
                                                                   --model_max_length 1560  \
                                                                   --learning_rate 1e-4  \
                                                                   --weight_decay 0.1  \
                                                                   --num_train_epochs 3  \
                                                                   --gradient_accumulation_steps 4  \
                                                                   --warmup_ratio 0.03  \
                                                                   --seed 1234  \
                                                                   --show_loss_step 10  \
                                                                   --lora_dim 16  \
                                                                   --lora_alpha 64  \
                                                                   --save_model_step 100  \
                                                                   --lora_dropout 0.1  \
                                                                   --output_dir /root/autodl-fs/text2sql_output_qlora  \
                                                                   --gradient_checkpointing  \
                                                                   --ds_file ds_zero2_no_offload.json  \
                                                                   --is_skip
```

## 模型融合
```shell
python3 merge_params.py --ori_model_dir "/root/autodl-fs/Qwen2_5_Coder_7B_Instruct/" --model_dir "/root/autodl-fs/text2sql_output_qlora/epoch-3-step-4221" --save_model_dir "/root/autodl-fs/Qwen2_5_Coder_7B_Instruct_text2sql/"
```