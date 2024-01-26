import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score

dic_list = ["result/gala_standard.json"]
with open("../chatglm/data/ID_num_dic.json") as read_file:
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
            if "yes" in data["predict"][0] or "Yes" in data["predict"][0] or "YES" in data["predict"][0]:
                pre.append(1)
            else:
                pre.append(0)
        result_list.append(average_precision_score(res, pre))
    print(f"{i}:map={sum(result_list)/len(result_list)}")