import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel
import json
data_dic = {}
with open("data/train_balance_2.json", "r") as read_file:
    all_lines = read_file.readlines()
instruction_dataset = InstructionDataset("data/pst_data")
# Initializes the model
#model = BaseModel.load("/root/huge_model/galactica/galactica")
model = BaseModel.create("galactica_lora")

# Finetuned the model
# model.finetune(dataset=instruction_dataset)
model.load("saved_model/")

# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))

# Save the model
# model.save("saved_model/")

write_file = open("result/galactica_lora_result.json", "w")
with open("../chatglm/data/test.json", "r") as read_file:
    all_lines = read_file.readlines()
for item in all_lines:
    dic = json.loads(item.strip())
    content = dic["content"]
    answer = dic["summary"]
    result = model.generate(texts=[content])
    result_dic = {"labels": answer, "predict":result}
    write_file.write(json.dumps(result_dic) + "\n")
    write_file.flush()
write_file.close()