# -*-coding:gbk -*-
import os
base_path = "v2/"
all_file = os.listdir(base_path)
for ktem in all_file:
    result = [0, 0, 0, 0] #"YesYes, YesNo, NoYes, NoNo"
    with open(base_path + ktem + "/generated_predictions.txt", "r") as read_file:
        all_lines = read_file.readlines()
        for item in all_lines:
             data = item.strip()
             data_list = data.split(",")
             if data_list[0][-2] == "s":
                 if data_list[1][-3] == "s":
                     result[0] += 1
                 else:
                     result[1] += 1
             else:
                 if data_list[1][-3] == "s":
                     result[2] += 1
                 else:
                     result[3] += 1
    Accuracy = (result[0]+result[3])/(sum(result))
    Precision = result[0]/(result[0]+result[2])
    Recall = result[0]/(result[0]+result[1])
    print(ktem+":")
    print("Accuracy:" + str(Accuracy))
    print("Precision:"+ str(Precision))
    print("Recall:"+ str(Recall))
    print("F1:"+ str((2*Precision*Recall)/(Precision+Recall)))
    print(result)
    
                 
                     