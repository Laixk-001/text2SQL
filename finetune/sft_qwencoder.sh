export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none
export PATH=/root/miniconda3/envs/qwen/bin:$PATH;

TRAIN_PATH=${1}
PRETRAINED_MODEL=${2}
OUTPUT_DIR=${3}
TEST_PATH=${4}

TRAIN_PATH=${TRAIN_PATH:-"/root/autodl-fs/DuSQL/text2sql_train_tokenizer_zh.json"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"/root/autodl-fs/Qwen2_5_Coder_7B_Instruct/"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/autodl-tmp/output_dir_qlora/lr${LR}-wr${WARMUP_STEPS}-wd${WEIGHT_DECAY}-bsz${BATCH_SIZE}-maxlen${MAX_LENGTH}/"}
TEST_PATH=${TEST_PATH:-"/root/autodl-fs/DuSQL/text2sql_dev_tokenizer_zh.json"}

GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-6105}
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
DEEPSPEED_CONFIG="default_offload_opt_param.json  "
BATCH_SIZE=1024
MICRO_BATCH_SIZE=4
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE))

LR=5e-5
MIN_LR=5e-6
WARMUP_STEPS=100
WEIGHT_DECAY=0.0
MAX_LENGTH=1280

echo $OUTPUT_DIR
echo "Pretrained Model" ${PRETRAINED_MODEL}
echo "WORLD_SIZE" $WORLD_SIZE "MICRO BATCH SIZE" $MICRO_BATCH_SIZE "GRAD_ACCU" $GRAD_ACCU
echo $DISTRIBUTED_ARGS

# cd ROOT_PATH="/path/to/sft/";
torchrun ${DISTRIBUTED_ARGS} train_qwen_1.py \
    --model_name_or_path  ${PRETRAINED_MODEL} \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --model_max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCU} \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "yes" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --report_to "tensorboard" \
    --bf16 True \
    --tf32 True \
    --truncate_source False