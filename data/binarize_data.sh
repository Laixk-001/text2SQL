# export PATH=/path/to/miniconda3/envs/qwen/bin:$PATH;
# cd ./Qwen2.5-Coder-evaluation/sft/;
cd /root/code/text2SQL/data/;
INPUT_PATH=${1}
OUTPUT_PATH=${2}
TOKENIZER_PATH=${3}
INPUT_PATH=${INPUT_PATH:-"/root/autodl-fs/DuSQL/text2sql_dev_text_zh.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"/root/autodl-fs/DuSQL/text2sql_dev_tokenizer_zh.json"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/root/autodl-fs/Qwen2_5_Coder_7B_Instruct/"}
python binarize_data.py -input_path ${INPUT_PATH} -output_path ${OUTPUT_PATH} -workers 64 -tokenizer_path ${TOKENIZER_PATH}