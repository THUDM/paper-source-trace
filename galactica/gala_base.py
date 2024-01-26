import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
import galai as gal
import json
from galai.notebook_utils import *

model = gal.load_model("standard")
with open("../chatglm/data/test.json", "r") as read_file:
    all_lines = read_file.readlines()
os.makedirs("result", exist_ok=True)
finished_num = 0
# with open("result/gala_standard.json") as read_file:
    # finished_num = len(read_file.readlines())
write_file = open("result/gala_standard.json", "a")
all_lines = all_lines[finished_num:]
for item in all_lines:
    this_line = json.loads(item)
    data = this_line["content"]
    label = this_line["summary"]
    data = data.replace(". The answer is [MASK].", "?")
    length = len(data)
    result = model.generate(data)[length:]
    write_file.write(json.dumps({"labels":label, "predict":result})+"\n")
    write_file.flush()
write_file.close()
