import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings('ignore')
import galai as gal
import json
from tqdm import tqdm
from galai.notebook_utils import *

model = gal.load_model("standard")
# with open("../chatglm/data/test.json", "r") as read_file:
    # all_lines = read_file.readlines()
os.makedirs("result", exist_ok=True)

write_file_2 = open("result/galactica_standard_4.json", "a")
# with open("../chatglm/data/test.json", "r") as read_file:
    # data_dic = json.load(read_file)
result_dic = {}
with open("data/test.jsonl", "r") as rf:
    for i, line in tqdm(enumerate(rf)):
        item = json.loads(line.strip())
        context  = item["context"]
        answer = item["label"]
        item_new = item
        if len(context) <= 3:
            item_new["predict"] = 0
        else:
            result = model.generate([" ".join(context.split()[-200:])])
            item_new["predict"] = result[0]
        write_file_2.write(json.dumps(item_new, ensure_ascii=False) + "\n")
        write_file_2.flush()

write_file_2.close()


"""
write_file = open("result/gala_standard.json", "w")
write_file_2 = open("result/gala_standard_2.json", "a")
with open("../chatglm/data/test.json", "r") as read_file:
    data_dic = json.load(read_file)
result_dic = {}
for item in tqdm(data_dic.keys()):
    data_list = data_dic[item]
    result_dic[item] = []
    for jtem in data_list:
        # dic = json.loads(jtem.strip())
        dic = jtem
        content = dic["content"]
        answer = dic["summary"]
        if len(content) <= 3:
            result = {"labels": answer, "predict": "No"}
        else:
            result = model.generate([" ".join(content.split()[-200:])])
            result = {"labels": answer, "predict":result[0]}
        result_with_id = {"id": item, "labels": answer, "predict":result["predict"]}
        write_file_2.write(json.dumps(result_with_id, ensure_ascii=False) + "\n")
        write_file_2.flush()
        result_dic[item].append(result)
json.dump(result_dic, write_file, indent=2)
write_file.close()
write_file_2.close()
"""
