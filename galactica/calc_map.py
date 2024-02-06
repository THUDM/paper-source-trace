import os
import json
from collections import defaultdict as dd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score


result_file = "galactica_lora_result_2.json"
pid_to_labels = dd(list)
pid_to_predict = dd(list)
with open(os.path.join("result", result_file), "r") as rf:
    for i, line in tqdm(enumerate(rf)):
        cur_item = json.loads(line.strip())
        pid = cur_item["pid"]
        cur_label = cur_item["label"]
        pid_to_labels[pid].append(cur_label)
        cur_predict = cur_item["predict"]
        try:
            s_idx = cur_predict.index("The answer is")
            cur_predict = cur_predict[s_idx + 13:]
        except:
            pass
        if "no" in cur_predict or "No" in cur_predict or "NO" in cur_predict:
            pid_to_predict[pid].append(0)
        else:
            pid_to_predict[pid].append(1)

maps = []
for pid in tqdm(pid_to_labels):
    cur_labels = pid_to_labels[pid]
    if sum(cur_labels) == 0:
        continue
    cur_predict = pid_to_predict[pid]
    cur_map = average_precision_score(cur_labels, cur_predict)
    maps.append(cur_map)

print(maps)
print(f"{i}:map={sum(maps)/len(maps)}")


"""
dic_list = ["result/gala_standard.json"]
for i in range(len(dic_list)):
    result_list = []
    with open(dic_list[i], "r") as read_file:
        result_dic = json.load(read_file)
    for key in result_dic.keys():
        pre = []
        res = []
        for jtem in result_dic[key]:
            # data = json.loads(jtem.strip())
            data = jtem
            if data["labels"] == "Yes":
                res.append(1)
            else:
                res.append(0)
            if "no" in data["predict"] or "No" in data["predict"] or "NO" in data["predict"]:
                pre.append(0)
            elif "yes" in data["predict"] or "Yes" in data["predict"] or "YES" in data["predict"]:
                pre.append(1)
            else:
                pre.append(0)
        if sum(res) == 0:
            continue
        cur_map = average_precision_score(res, pre)
        print(cur_map)
        result_list.append(cur_map)
    print(f"{i}:map={sum(result_list)/len(result_list)}")
"""