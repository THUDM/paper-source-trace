import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
# base_path = "/data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/test/v2/"
base_path = "/data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/test/finetune/"
num = 5
dic_list = [base_path + str(i) + "000/generated_predictions.txt" for i in range(1, 6)]
with open("ID_num_dic.json") as read_file:
    trans_dic = json.load(read_file)
for i in range(len(dic_list)):
    result_list = []
    result_dic = {}
    with open(dic_list[i], "r") as read_file:
        all_lines = read_file.readlines()
    for item in trans_dic.keys():
        this_list = trans_dic[item]
        pre = []
        res = []
        for jtem in this_list:
            data = json.loads(all_lines[jtem].strip())
            if data["labels"] == "Yes":
                res.append(1)
            else:
                res.append(0)
            #if data["predict"] == "Yes":
            #    pre.append(1)
            #else:
            #    pre.append(0)
            if len(data["predict"]) == 0:
                pre.append(0)
            else:
                pre.append(1)
        result_list.append(average_precision_score(res, pre))
    print(f"{i}:map={sum(result_list)/len(result_list)}")