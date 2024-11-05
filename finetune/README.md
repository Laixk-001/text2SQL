
- 单机四卡训练
```shell
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 5545 train.py --train_path /root/auto-fs/DuSQL/text2sql_train_tokenizer_zh.json  \
                                                                   --test_path /root/auto-fs/DuSQL/text2sql_dev_tokenizer_zh.json  \
                                                                   --model_name_or_path /root/autodl-fs/Qwen2_5_Coder_7B_Instruct/  \
                                                                   --per_device_train_batch_size 2  \
                                                                   --max_len 2048  \
                                                                   --max_src_len 1560  \
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
                                                                   --output_dir ./output_dir_qlora  \
                                                                   --gradient_checkpointing  \
                                                                   --ds_file ds_zero2_no_offload.json  \
                                                                   --is_skip
```
