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
  - 