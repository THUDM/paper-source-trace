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
write_file = open("result/gala_standard.json", "w")
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
            result = {"labels": answer, "predict":result}
        result_dic[item].append(result)
json.dump(result_dic, write_file, indent=2)
write_file.close()
