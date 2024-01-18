import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel
import json
data_dic = {}
with open("datasets/train_balance_2.json", "r") as read_file:
    all_lines = read_file.readlines()
instruction_dataset = InstructionDataset("alpaca_data")
# Initializes the model
#model = BaseModel.load("/root/huge_model/galactica/galactica")
model = BaseModel.create("galactica")
# Finetuned the model
model.finetune(dataset=instruction_dataset)
# Once the model has been finetuned, you can start doing inferences
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
# Save the model
model.save("/root/huge_model/galactica/saved_model/")