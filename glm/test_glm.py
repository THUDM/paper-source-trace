import os
from os.path import join
from collections import defaultdict as dd
import numpy as np
import torch
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from sklearn.metrics import average_precision_score

import utils
import settings


def eval_test(model_name, ckpt_epoch=1, year=2023, role="test", ft=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    out_dir = "glm/saved"
    prefix = model_name.split("/")[1].lower()
    model_path = os.path.join(out_dir, "{}-epoch-{}.pt".format(prefix, ckpt_epoch))
    if not ft:
        model_infer = AutoModelForMultipleChoice.from_pretrained(model_name, trust_remote_code=True)
        # model_infer = AutoModelForMultipleChoice.from_pretrained("/home/zhangfanjin/.cache/huggingface/hub/models--THUDM--glm-10b/snapshots/696788d4f82ac96b90823555f547d1e754839ff4", trust_remote_code=True)
        model_infer.to('cuda')
    elif prefix == "glm-10b":
        model_infer = torch.load(model_path, map_location=torch.device('cuda'))
    elif prefix == "glm-2b":
        model_infer = torch.load(model_path, map_location=torch.device('cuda'))
        # model_infer.to('cuda')
    model_infer.eval()
    print("model load successfully")

    papers_test = utils.load_json(join(settings.DATA_TRACE_DIR, str(year)), "paper_source_trace_{}.json".format(role))
    pids_test = {p["_id"] for p in papers_test}

    truths = papers_test
    pid_to_source_titles = dd(list)
    for paper in tqdm(truths):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    xml_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    candidates = ["No", "Yes"]
    metrics = []
    f_idx = 0

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

        contexts_sorted = ["The context is: " + " ".join(bib_to_contexts[bib]) 
                            + ". Is the current reference important? Please answer Yes or No. The answer is [MASK]." 
                            for bib in bib_sorted]
        contexts_sorted = [x[-500:] for x in contexts_sorted]

        predicted_scores = []
        for cur_context in contexts_sorted:
            token = tokenizer([cur_context], return_tensors="pt", padding=True)
            inputs = tokenizer.build_inputs_for_multiple_choice(token, [candidates])
            inputs = inputs.to('cuda')
            outputs = model_infer(**inputs)
            logits = outputs.logits
            # print("logits", logits.shape)
            score = logits.detach().cpu().numpy().tolist()
            # print("score", score)
            predicted_scores.append(score[0][1])

        try:
            for ii in range(len(predicted_scores)):
                bib_idx = int(bib_sorted[ii][1:])
                y_score[bib_idx] = predicted_scores[ii]
        except IndexError as e:
            metrics.append(0)
            continue

        cur_map = average_precision_score(y_true, y_score)
        metrics.append(cur_map)
        f_idx += 1
        if f_idx % 20 == 0:
            print("map until now", np.mean(metrics), len(metrics), cur_map)

    map_avg = np.mean(metrics)
    print("epoch {} average map".format(ckpt_epoch), map_avg, len(metrics))
    return np.mean(metrics)


if __name__ == "__main__":
    # eval_test(model_name="THUDM/GLM-2b", ckpt_epoch=1)

    """
    model_name = "THUDM/GLM-2b"
    prefix = model_name.split("/")[1].lower()
    wf = open("glm/saved/valid_map_{}.txt".format(prefix), "w")
    for i in range(10):
        cur_map = eval_test(model_name=model_name, ckpt_epoch=i, role="valid")
        wf.write("{}\t{}\n".format(i, cur_map))
        wf.flush()
    wf.close()
    """
    
    # eval_test(model_name="THUDM/GLM-2b", ckpt_epoch=3)
    eval_test(model_name="THUDM/GLM-10b", ckpt_epoch=1)
