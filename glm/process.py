import os
from os.path import join
import json
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
import numpy as np
from fuzzywuzzy import fuzz

import utils
import settings


def prepare_train_test_data_for_glm():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []

    truths = utils.load_json(settings.DATA_TRACE_DIR, "paper_source_trace_2022_final_filtered.json")
    pid_to_source_titles = dd(list)
    for paper in tqdm(truths):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    papers_train = utils.load_json(settings.DATA_TRACE_DIR, "paper_source_trace_train.json")
    papers_valid = utils.load_json(settings.DATA_TRACE_DIR, "paper_source_trace_valid.json")
    papers_test = utils.load_json(settings.DATA_TRACE_DIR, "paper_source_trace_test.json")

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}
    pids_test = {p["_id"] for p in papers_test}

    in_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    files = sorted(files)
    for file in tqdm(files):
        f = open(join(in_dir, file), encoding='utf-8')
        cur_pid = file.split(".")[0]
        if cur_pid not in pids_train and cur_pid not in pids_valid and cur_pid not in pids_test:
            continue
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

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
        
        flag = False

        cur_pos_bib = set()

        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    cur_pos_bib.add(bid)
        
        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        
        if not flag:
            continue
    
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        bib_to_contexts = utils.find_bib_context(xml)

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        if cur_pid in pids_train:
            cur_x = x_train
            cur_y = y_train
        elif cur_pid in pids_valid:
            cur_x = x_valid
            cur_y = y_valid
        elif cur_pid in pids_test:
            cur_x = x_test
            cur_y = y_test
        else:
            continue
            # raise Exception("cur_pid not in train/valid/test")
        
        for bib in cur_pos_bib:
            cur_context = "The context is: " + " ".join(bib_to_contexts[bib]) + ". Is the current reference important? Please answer Yes or No. The answer is [MASK]."
            cur_x.append(cur_context)
            cur_y.append(1)
    
        for bib in cur_neg_bib_sample:
            cur_context = "The context is: " + " ".join(bib_to_contexts[bib]) + ". Is the current reference important? Please answer Yes or No. The answer is [MASK]."
            cur_x.append(cur_context)
            cur_y.append(0)

    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid), "len(x_test)", len(x_test))

    out_dir = "glm/data/"
    os.makedirs(out_dir, exist_ok=True)

    with open(join(out_dir, "train.json"), "w") as f:
        for i in range(len(x_train)):
            f.write(json.dumps({"inputs_pretokenized": x_train[i], "choices_pretokenized": ["No", "Yes"], "label": y_train[i]}) + "\n")
    

    with open(join(out_dir, "valid.json"), "w") as f:
        for i in range(len(x_valid)):
            f.write(json.dumps({"inputs_pretokenized": x_valid[i], "choices_pretokenized": ["No", "Yes"], "label": y_valid[i]}) + "\n")

    with open(join(out_dir, "test.json"), "w") as f:
        for i in range(len(x_test)):
            f.write(json.dumps({"inputs_pretokenized": x_test[i], "choices_pretokenized": ["No", "Yes"], "label": y_test[i]}) + "\n")


if __name__ == "__main__":
    prepare_train_test_data_for_glm()
