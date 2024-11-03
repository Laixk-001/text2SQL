# text2SQL
## 项目简介

通过Qwen2-7B模型实现SQL语句生成，包括Qwen2-7B模型的微调，DPO对齐和测试。

项目主要结构如下：
- data：存放数据构造的文件
  - dusql_process.py: 使用dusql数据集构造微调数据集和测试数据集
  - translation_service.py：部署翻译模型
- finetune: 微调相关文件
  - cpp_kernels.py
  - train.py
- src: 源文件
  - utils.py
- merge: 合并模型
  - merge_QLoRA.py
- predict: 预测
  - predict.py


## 数据处理

数据预处理需要运行dusql_process.py文件，会在data文件夹中生成训练集和测试集文件。

命令如下：

```shell
cd data

python3 dusql_process.py
```
本次微调主要针对[dusql数据](https://aistudio.baidu.com/competition/detail/47/0/task-definition) 进行应用，并且由于当前dusql数据中，表格信息以中文为主，因此本次我们还将采用翻译模型对数据中的表格字段信息进行翻译，翻译器可以使用开源中-英翻译模型。

本项目中采用huggiFace中的 [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)

## 模型微调

模型训练需要运行train.py文件，会自动生成output_dir文件夹，存放每个save_model_step保存的模型文件。

- 单机四卡训练
需要进到源码内：/root/miniconda3/envs/text2sql/lib/python3.10/site-packages/deepspeed/ops/op_builder
修改builder.py文件：
  if sys_cuda_version != torch_cuda_version:
    return True

```shell
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 5545 train_qwen.py --train_path /root/autodl-fs/DuSQL/text2sql_train_zh.json  \
                                                                   --test_path /root/autodl-fs/DuSQL/text2sql_dev_zh.json  \
                                                                   --model_name_or_path /root/autodl-fs/Qwen2_5_Coder_7B_Instruct  \
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
                                                                   --output_dir /root/autodl-tmp/output_dir_qlora  \
                                                                   --gradient_checkpointing  \
                                                                   --ds_file default_offload_opt_param.json  
```

## 模型推理

模型融合执行命令：
```shell
python3 merge_params.py --ori_model_dir "/root/autodl-fs/Qwen-1_8B-Chat/" --model_dir "/root/auto-fs/DiaExtra/output_dir_qlora/epoch_1" --save_model_dir "/root/auto-fs/DiaExtra/output_dir_qlora/epoch_1"
```

推理执行命令：
```shell
cd predict
python3 predict.py --model_path "your_model_path"
```

