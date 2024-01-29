import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
# base_path = "/data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/test/v2/"
# base_path = "output/ptuning/"
base_path = "output/finetune/"

num = 5
dic_list = [base_path + "generated_predictions.txt" for i in range(1, 2)]
# dic_list = [base_path + str(i) + "000/generated_predictions.txt" for i in range(1, 6)]
for i in range(len(dic_list)):
    result_list = []
    with open(dic_list[i], "r") as read_file:
        result_dic = json.load(read_file)
    for key in result_dic.keys():
        pre = []
        res = []
        for jtem in result_dic[key]:
            data = json.loads(jtem.strip())
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