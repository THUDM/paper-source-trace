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
# import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict as dd
from os.path import join
import os
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

import utils
import settings

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

    dataset.save_to_disk(str("galactica/data/pst_data")) 


def make_balance_data_for_galactica():
    target_list = ["train", "valid"]
    yes_list, no_list = [], []
    for item in target_list:
        with open("glm/data/" + item + '.json', "r") as read_file:
            all_lines = read_file.readlines()
            for data in all_lines:
                data_dic = json.loads(data.strip())
                if data_dic["label"] == 1:
                    yes_list.append(data_dic)
                else:
                    no_list.append(data_dic)
        np.random.seed(42)
        no_list = np.random.choice(no_list, len(yes_list), replace=False).tolist()
        all_list = yes_list+no_list
        print(len(all_list))
        np.random.shuffle(all_list)
        with open("galactica/data/" + item+"_balance_gala.json", "w") as write_file:
            for jtem in all_list:
                new_item = {}
                new_item["output"] = "Yes" if jtem["label"] else "No"
                new_item["instruction"] = jtem["inputs_pretokenized"][:-7]
                new_item["input"] = ""
                write_file.write(json.dumps(new_item)+"\n")


def gen_test_data_json_lines(year=2023):
    data_year_dir = join(settings.DATA_TRACE_DIR, str(year))
    papers_test = utils.load_json(data_year_dir, "paper_source_trace_test.json")
    pids_test = {p["_id"] for p in papers_test}

    in_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        cur_pid = f.split(".")[0]
        if f.endswith(".xml") and cur_pid in pids_test:
            files.append(f)

    truths = papers_test
    pid_to_source_titles = dd(list)
    for paper in tqdm(truths):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())


    xml_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    wf = open("galactica/data/test.jsonl", "w")

    for paper in tqdm(papers_test):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".tei.xml")
        f = open(file, encoding='utf-8')

        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        bib_to_contexts = utils.find_bib_context(xml)
        bib_sorted = sorted(bib_to_contexts.keys())

        for bib in bib_sorted:
            cur_bib_idx = int(bib[1:])
            if cur_bib_idx + 1 > n_refs:
                n_refs = cur_bib_idx + 1
        
        y_true = [0] * n_refs
        y_score = [0] * n_refs

        flag = False
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    b_idx = int(bid[1:])
                    y_true[b_idx] = 1
        
        if not flag:
            continue

        contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        # print(bid_to_title)

        for bi, cur_bib in enumerate(bib_sorted):
            new_item = {"pid": cur_pid, "bib_id": cur_bib}
            cur_context = contexts_sorted[bi]
            cur_context = cur_context + ". Is the current reference important? Please answer Yes or No. The answer is "
            cur_label = y_true[int(cur_bib[1:])]
            new_item["label"] = cur_label
            new_item["context"] = cur_context
            try:
                new_item["title"]= bid_to_title[cur_bib]
            except:
                pass
            wf.write(json.dumps(new_item) + "\n")
            wf.flush()

    wf.close()


# preprocess_alpaca_json_data("data/train_balance_2.json")

# make_balance_data_for_galactica()
# preprocess_alpaca_json_data("galactica/data/train_balance_gala.json")
gen_test_data_json_lines(2023)