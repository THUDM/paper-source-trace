import json
import random

target_list = ["train", "valid"]
yes_list, no_list = [], []
for item in target_list:
    with open(item + '.json', "r") as read_file:
        all_lines = read_file.readlines()
        for data in all_lines:
            data_dic = json.loads(data.strip())
            if data_dic["summary"] == "Yes":
                yes_list.append(data.strip())
            else:
                no_list.append(data.strip())
    no_list = random.sample(no_list, len(yes_list))
    all_list = yes_list+no_list
    random.shuffle(all_list)
    with open(item+"_balance.json", "w") as write_file:
        for jtem in all_list:
            write_file.write(jtem+"\n")
            
        