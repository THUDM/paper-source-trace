import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel
import json
from tqdm import tqdm

data_dic = {}
# with open("data/train_balance_2.json", "r") as read_file:
    # all_lines = read_file.readlines()
instruction_dataset = InstructionDataset("data/pst_data")
# Initializes the model
#model = BaseModel.load("/root/huge_model/galactica/galactica")
# model = BaseModel.create("galactica_lora")
model = BaseModel.create("galactica")

# Finetuned the model
# model.finetune(dataset=instruction_dataset)
# model.load("saved_model/")

# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))

# Save the model
# model.save("saved_model/")

# write_file = open("result/galactica_lora_result.json", "w")
# write_file_2 = open("result/galactica_lora_result_2.json", "a")
write_file_2 = open("result/galactica_standard_3.json", "a")
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
            result = model.generate(texts=[" ".join(context.split()[-200:])])
            item_new["predict"] = result[0]
        write_file_2.write(json.dumps(item_new, ensure_ascii=False) + "\n")
        write_file_2.flush()

write_file_2.close()

"""
for item in tqdm(data_dic.keys()):
    data_list = data_dic[item]
    result_dic[item] = []
    for jtem in data_list:
        dic = jtem
        content = dic["content"]
        answer = dic["summary"]
        if len(content) <= 3:
            result = {"labels": answer, "predict": "No"}
        else:
            result = model.generate(texts=[content])
            result = {"labels": answer, "predict":result[0]}
        result_with_id = {"id": item, "labels": answer, "predict":result["predict"]}
        write_file_2.write(json.dumps(result_with_id, ensure_ascii=False) + "\n")
        result_dic[item].append(result)
json.dump(result_dic, write_file, indent=2)
write_file.close()
write_file_2.close()
"""