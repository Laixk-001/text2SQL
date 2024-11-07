import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Service:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).cuda()
    
    def predict(self, sql_info, query):
        """
        sql_info: 建表语句
        query: 用户问题
        """
        messages = [
            {'role':'system','content':"### Instructions:\nYour task is convert a question into a SQL query, given a Postgres database schema.\nAdhere to these rules:\n- **Deliberately go through the question and database schema word by word** to appropriately answer the question\n- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.\n- When creating a ratio, always cast the numerator as float"},
            {'role':'user','content': f"### Input:\nGenerate a SQL query that answers the question `{query}`.\nThis query will run on a database whose schema is represented in this string:\n{sql_info}"}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            self.model.device
        )
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95,
                                      num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id)
        result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return result

def parse_args():
    parser = argparse.ArgumentParser(description='train bpe tokenizer')
    parser.add_argument('--model_path', type=str, default="", help='保存的模型路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ss = Service(args.model_path)
    query = "没有比赛记录的篮球运动员有哪些，同时给出他们在球场上位于哪个位置？"
    sql_info = "CREATE TABLE 篮球运动员(\n  词条id VARCHAR(50) PRIMARY KEY, -- ['Phrase_id']\n  中文名 INTEGER, -- ['Chinese_Name']\n  场上位置 VARCHAR(50), -- ['Field_position']\n  球队 VARCHAR(50), -- ['Team']\n  年龄 VARCHAR(50), -- ['Age']\n);\nCREATE TABLE 比赛记录(\n  赛季 INTEGER PRIMARY KEY, -- ['Season']\n  球队 INTEGER, -- ['Team']\n  赛事类型 VARCHAR(50), -- ['Type_of_event']\n  球员id VARCHAR(50), -- ['Player_id']\n  出场次数 INTEGER, -- ['Number_of_appearances']\n  首发次数 INTEGER, -- ['Number_of_initial_issuances']\n  投篮 INTEGER, -- ['Shoot_it']\n  罚球 INTEGER, -- ['Strike!']\n  三分球 INTEGER, -- ['Three']\n  总篮板 INTEGER, -- ['Total_basket']\n  抢断 INTEGER, -- [\"I'll_take_it\"]\n  助攻 INTEGER, -- ['Accompaniment']\n  防守 INTEGER, -- ['Defense']\n  犯规 INTEGER, -- [\"It's_a_foul\"]\n  得分 INTEGER, -- ['Score']\n);\nCREATE TABLE 生涯之最(\n  球员id INTEGER PRIMARY KEY, -- ['Player_id']\n  单场得分 INTEGER, -- ['Single_score']\n  篮板球次数 INTEGER, -- ['Number_of_basketballs']\n  抢断次数 INTEGER, -- ['Number_of_breakouts']\n  助攻次数 INTEGER, -- ['Number_of_offensives']\n  盖帽次数 INTEGER, -- ['Number_of_caps']\n  比赛时间 INTEGER, -- ['Game_time']\n  比赛对手 DATETIME, -- ['Fighter']\n);\n-- 比赛记录.球员id can be joined with 篮球运动员.篮球运动员\n-- 生涯之最.球员id can be joined with 篮球运动员.篮球运动员"
    response = ss.predict(query,sql_info)
    print(response)