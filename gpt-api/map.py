import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
# base_path = "/data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/test/v2/"
base_path = "/data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/test/finetune/"
num = 5
dic_list = ["result/gpt-3.5-turbo2.json", "result/gpt-4.json"]
for i in range(len(dic_list)):
    result_list = []
    with open(dic_list[i], "r") as read_file:
        result_dic = json.load(read_file)
    for item in result_dic.keys():
        this_list = result_dic[item]
        pre = []
        res = []
        for jtem in this_list:
            if jtem["labels"] == "Yes":
                res.append(1)
            else:
                res.append(0)
            #if data["predict"] == "Yes":
            #    pre.append(1)
            #else:
            #    pre.append(0)
            if "yes" in jtem["predict"] or "Yes" in jtem["predict"]:
                pre.append(1)
            elif "not important" in jtem["predict"]:
                pre.append(0)
            elif "important" in jtem["predict"]:
                pre.append(1)
            else:
                pre.append(0)
        result_list.append(average_precision_score(res, pre))
    print(f"{i}:map={sum(result_list)/len(result_list)}")