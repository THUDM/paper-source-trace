#import json 
#import jsonlines
#with open("datasets/train_balance.json", "r") as read_file:
#    all_lines = read_file.readlines()
#write_file = jsonlines.open("datasets/train_balance_3.jsonl", "w")
#for i, item in enumerate(all_lines):
#    data_dic = json.loads(item.strip())
#    new_dic = {"id":f"seed_task_{i}", "name":f"{i}", "instruction":data_dic["content"], "instances":[{"input":"", "output":data_dic["summary"], "is_classification": False}]}
#    write_file.write(json.dumps(new_dic))
import json

from datasets import Dataset, DatasetDict

# Convert the alpaca JSON dataset to HF format


# Right now only the HuggingFace datasets are supported, that's why the JSON Alpaca dataset
# needs to be converted to the HuggingFace format. In addition, this HF dataset should have 3 columns for instruction finetuning: instruction, text and target.
def preprocess_alpaca_json_data(alpaca_dataset_path: str):
    """Creates a dataset given the alpaca JSON dataset. You can download it here: https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

    :param alpaca_dataset_path: path of the Alpaca dataset
    """
    read_file = open(alpaca_dataset_path, "r")
    instructions = []
    inputs = []
    outputs = []
    for item in read_file:
        dic = json.loads(item.strip())
        instructions.append(dic["instruction"])
        inputs.append(dic["input"])
        outputs.append(dic["output"])

    data_dict = {
        "train": {"instruction": instructions, "text": inputs, "target": outputs}
    }

    dataset = DatasetDict()
    # using your `Dict` object
    for k, v in data_dict.items():
        dataset[k] = Dataset.from_dict(v)

    dataset.save_to_disk(str("/root/huge_model/galactica/alpaca_data")) 
preprocess_alpaca_json_data("datasets/train_balance_2.json")