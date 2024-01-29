import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
import galai as gal
import json
from galai.notebook_utils import *

model = gal.load_model("standard")
with open("../chatglm/data/test.json", "r") as read_file:
    all_lines = read_file.readlines()
os.makedirs("result", exist_ok=True)
write_file = open("result/gala_standard.json", "w")
with open("../chatglm/data/test.json", "r") as read_file:
    data_dic = json.load(read_file)
result_dic = {}
for item in data_dic.keys():
    data_list = data_dic[item]
    result_dic[item] = []
    for jtem in data_list:
        dic = json.loads(jtem.strip())
        content = dic["content"]
        answer = dic["summary"]
        result = model.generate(texts=[content])
        result = {"labels": answer, "predict":result}
        result_dic[item].append(result)
json.dump(result_dic, write_file, indent=2)
write_file.close()
