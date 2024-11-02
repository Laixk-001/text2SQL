# -*- coding: utf-8 -*-
import argparse
import json
import os
import requests
from tqdm import tqdm
import copy
from translation_service import Translation

import jsonlines
import os
import numpy as np
import transformers
import sys
from typing import Dict
import argparse
import itertools
import json

"""
DuSQL数据库信息:
    column_names: 表格中的表字段信息
    column_types: 表字段的数据类型
    db_id: 数据库名称
    foreign_keys: 外键
    primary_keys: 主键
    table_names: 表格名称
"""

class DusqlDataSet:
    """
    获得建表语句，例子：
    CREATE TABLE 水果(
        词条 id VARCHAR(50) PRIMARY KEY, -- Phrasing id
        名称 INTEGER, -- Name
        特性 VARCHAR(50), --Features
        每100克水分 INTEGER, -- Every 100 grams
    )
    """
    def __init__(self, home_path, translation_model_path):
        self.home_path = home_path
        self.translation = Translation(translation_model_path)
    
    def translation_service(self, text):
        result = self.translation.translate(text)
        en_query = [r.strip(".").replace(".", " ").replace(",", " ") for r in result]
        en_query = [r.replace(" ", "_") for r in en_query]
        return en_query
    
    @staticmethod
    def load_data(path):
        with open(path,'r',encoding='utf-8')as f:
            data = json.load(f)
        return data
    
    # 获得每一列数据的属性
    def get_column_types(self, col_type):
        type_dict = {"number": "INTEGER", "text": "VARCHAR(50)", "binary": "BINARY", "time": "DATETIME",
                     "data": "DATETIME"}
        
        if col_type in type_dict:
            return type_dict[col_type]
        return "VARCHAR(50)"
    
    # 构建sql语言
    def get_sqlite(self):
        result = {}
        with open(os.path.join(self.home_path, "new_schema.jsonl"), "r", encoding="utf-8") as f:
        # with open(os.path.join(self.home_path, "db_schema.json"), "r", encoding="utf-8") as f:
            for line in f:
                whole_sql_info = []
                sample = json.loads(line)
                db_id = sample["db_id"]
                columns_en = sample["column_en"]
                table_en = sample['table_en']
                joined_info = sample['joined_info']    
                for table_name, columns in sample["table_info"].items():
                    is_first = True
                    table_info  = f"CREATE TABLE {table_name}"
                    column_info = []
                    for column in columns:
                        column_name_zh, column_type = column
                        column_name_en = columns_en[column_name_zh]
                        # column_name_en = " ".join(column_name_en.split("_"))
                        column_sql_type = self.get_column_types(column_type)
                        if is_first:
                            """定义为主键"""
                            column_info.append(f"  {column_name_zh} {column_sql_type} PRIMARY KEY, -- {column_name_en}")
                            is_first = False
                        else:
                            column_info.append(f"  {column_name_zh} {column_sql_type}, -- {column_name_en}")
                    one_table_info = table_info + "(\n" + "\n".join(column_info) + "\n);"
                    whole_sql_info.append(one_table_info)
                joined_part = []
                """表关联"""
                for one_join in joined_info:
                    a, b = one_join
                    table_name_zh_a, column_name_zh_a = a[0], a[1]
                    table_name_zh_b, column_name_zh_b = b[0], b[1]
                    one_join_info = f"-- {table_name_zh_a}.{column_name_zh_a} can be joined with {table_name_zh_b}.{table_name_zh_b}"
                    joined_part.append(one_join_info)
                whole_sql_info.append("\n".join(joined_part))
                result[db_id] = {"sqlite": "\n".join(whole_sql_info), "columns_en": columns_en, "table_en": table_en}
        return result
    
    def trans_schema(self):
        db_schema = self.load_data(os.path.join(self.home_path, "db_schema.json"))
        new_schema = []
        for one_db in tqdm(db_schema):
            db_id = one_db['db_id']
            table_info = {}
            column_en = {}
            table_en = {}
            for i, column_info in enumerate(tqdm(one_db['column_names'][1:])):
                table_id, column_name = column_info
                column_name_en = self.translation_service(column_name)
                column_en[column_name] = column_name_en
                column_type = one_db['column_types'][i]
                table_name = one_db['table_names'][table_id]
                table_name_en = self.translation_service(table_name)
                table_en[table_name] = table_name_en
                if table_name in table_info:
                    table_info[table_name].append([column_name, column_type])
                else:
                    table_info[table_name] = [[column_name, column_type]]
            foreign_keys = one_db["foreign_keys"]
            joined_info = []
            for keys in foreign_keys:
                a, b = one_db['column_names'][keys[0]], one_db['column_names'][keys[1]]
                table_name_a, column_name_a = one_db['table_names'][a[0]], a[1]
                table_name_b, column_name_b = one_db['table_names'][b[0]], b[1]
                joined_info.append(([table_name_a, column_name_a], [table_name_b, column_name_b]))
            schema_info = {"db_id": db_id, "table_info": table_info, "joined_info": joined_info,
                           "column_en": column_en, "table_en": table_en}
            new_schema.append(schema_info)
        return new_schema
    
def make_llm_data(home_path, file_name, save_name,tokenizer, sqlite_info_name="sqlite_info_zh.json"):
    llm_data = []
    with open(os.path.join(home_path, sqlite_info_name), "r", encoding="utf-8")as f:
        sqlite_info = json.load(f)
    with open(os.path.join(home_path, file_name), "r", encoding="utf-8")as f:
        samples = json.load(f)
        IGNORE_INDEX = tokenizer.pad_token_id
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        
        im_start = tokenizer("<|im_start|>").input_ids[0]
        im_end = tokenizer("<|im_end|>").input_ids[0]
        nl_tokens = tokenizer('\n').input_ids
        if len(nl_tokens) > 0:
            nl_tokens = nl_tokens[-1:]
        
        _system = tokenizer('system').input_ids + nl_tokens
        _user = tokenizer('user').input_ids + nl_tokens
        _assistant = tokenizer('assistant').input_ids + nl_tokens
        objs = []

        for sample in tqdm(samples):
            db_id = sample["db_id"]
            question = sample["question"]
            sql_query_zh = sample["query"]
            sqlite_query = sqlite_info[db_id]["sqlite"]
            # sqlite_query = sample["sql"]

            input_id, target, test_input_ids = [], [], []
            system_message = "You are a helpful assistant."
            
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            test_input_ids += system
            target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
            assert len(input_id) == len(target), "Input and target lengths do not match."

            input_role_user = '<|im_start|>user'
            sentence_input_user = f'### Input:\nGenerate a SQL query that answers the question `{question}`.\nThis query will run on a database whose schema is represented in this string:\n{sqlite_query}'
            _input_id = tokenizer(input_role_user).input_ids + nl_tokens + tokenizer(sentence_input_user, add_special_tokens=False).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            test_input_ids += _input_id

            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            target += _target

            input_role_assistant = '<|im_start|>assistant'
            sentence_input_assistant = f'### Response:\nBased on your instructions, here is the SQL query I have generated to answer the question `{question}`:\n`{sql_query_zh}`'
            _input_id = tokenizer(input_role_user).input_ids + nl_tokens + tokenizer(sentence_input_assistant, add_special_tokens=False).input_ids + [im_end] + nl_tokens
            input_id += _input_id

            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(input_role_assistant).input_ids) + _input_id[len(tokenizer(input_role_assistant).input_ids) + 1:-2] + [im_end] + nl_tokens

            test_input_ids += tokenizer(input_role_assistant).input_ids + nl_tokens

            objs.append(dict(
                test_input_ids=test_input_ids,
                input_ids=input_id,
                label=target,
            ))

    with open(os.path.join(home_path, save_name),"w",encoding="utf-8")as fout:
        fout.writelines("\n".join([json.dumps(one, ensure_ascii=False) for one in objs]))

    # Set special tokens globally to avoid adding them multiple times.
def setup_tokenizer(tokenizer):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|repo_name|>",
            "<|file_sep|>", "<|im_start|>", "<|im_end|>"
        ]
    })
    return tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='llama2-7B QLoRA')
    parser.add_argument('--sql_home_path', type=str, default="./dusql/", help='dusql数据保存地址')
    parser.add_argument('--translation_model_path', type=str, default="", help='翻译模型地址')

    parser.add_argument('--input_path', '-input_path', type=str, default="sft.jsonl", help='Path to input file')
    parser.add_argument('--output_path', '-output_path', type=str, default="sft.jsonl", help='Path to output file')
    parser.add_argument('--workers', '-workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--chunk_size', '-chunk_size', type=int, default=10240, help='Chunk size for file processing')
    parser.add_argument('--max_len', '-max_len', type=int, default=32768, help='Maximum length for tokenization')
    parser.add_argument('--tokenizer_path', '-tokenizer_path', type=str, default="/root/autodl-fs/Qwen2_5_Coder_7B_Instruct", help='Path to tokenizer')
    return parser.parse_args()

if __name__ == "__main__":
    home_path = "/root/autodl-fs/DuSQL/"
    translation_model_path = "/root/autodl-fs/opus-mt-zh-en"
    tokenizer_path = "/root/autodl-fs/Qwen2_5_Coder_7B_Instruct"
    # data = DusqlDataSet(home_path, translation_model_path)
    # new_schema = data.trans_schema()
    # with open(os.path.join(home_path, "new_schema.jsonl"), "w", encoding = "utf-8")as f:
    #     for n in new_schema:
    #         f.write(json.dumps(n) + '\n')
    # result = data.get_sqlite()
    # with open(os.path.join(home_path, "sqlite_info_zh.json"), "w", encoding="utf-8") as f:
    #     json.dump(result, f,ensure_ascii=False)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        add_bos_token=False,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        cache_dir=None,
        model_max_length=8192 * 4,
        truncation=True,
        padding_side="left",
        trust_remote_code=True
    )
    tokenizer = setup_tokenizer(tokenizer)  # Set special tokens once
        
    make_llm_data(home_path,"dev.json", "text2sql_dev_zh.json",tokenizer)
    make_llm_data(home_path,"train.json", "text2sql_train_zh.json",tokenizer)