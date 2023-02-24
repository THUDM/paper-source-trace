import re
import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.metrics import average_precision_score

import utils
import settings


def extract_one_paper_via_rule(xml):
    bs = BeautifulSoup(xml, "xml")
    ref = []
    importantlist = []
    for item in bs.find_all(type='bibr'):
        if "target" not in item.attrs:
            continue
        item_str = "<ref type=\"bibr\" target=\"{}\">{}</ref>".format(item.attrs["target"], item.get_text())
        try:
            refer = item.attrs["target"][1:]
            ref.append((item_str, refer))    # 找到上下文
            # print(refer)
            pass
        except IndexError as e:
            continue
    xml = xml.lower()
    s2 = [ii for ii in range(len(xml)) if xml.startswith('motivated by', ii)]
    s3 = [ii for ii in range(len(xml)) if xml.startswith('inspired by', ii)]
    s = s2 + s3
    pos_to_signal = {}
    for i in s2:
        pos_to_signal[i] = "motivated by"
    for i in s3:
        pos_to_signal[i] = "inspired by"

    for i in ref:
        cur_bibr, idx = i
        p_ref = [ii for ii in range(len(xml)) if xml.startswith(cur_bibr, ii)]
        # print("p_ref", p_ref)
        for j in p_ref:
            for k in s:
                if abs(j-k) < 100:
                    importantlist.append(idx)
                    # print("hit***************************", j, k, i, pos_to_signal[k])
                    break
    return importantlist


def find_paper_source_by_rule():
    truths = utils.load_json(settings.DATA_TRACE_DIR, "paper_source_trace_test.json")
    pid_to_source_titles = dd(list)
    for paper in tqdm(truths):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())
    xml_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    metrics = []
    p_idx = 0

    for paper in tqdm(truths):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".tei.xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

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

        y_true = []
        y_score = []
        try:
            source_titles = pid_to_source_titles[cur_pid]
            if len(source_titles) == 0:
                print("hit1")
                raise
                continue
            pred_sources = extract_one_paper_via_rule(xml)
            y_true = [0]* n_refs
            y_score = [0]* n_refs

            for bid in bid_to_title:
                cur_ref_title = bid_to_title[bid]
                for label_title in source_titles:
                    if fuzz.ratio(cur_ref_title, label_title) >= 80:
                        b_idx = int(bid[1:])
                        y_true[b_idx] = 1
                        break
            for ii in pred_sources:
                y_score[int(ii[1:])] = 1
            if sum(y_true) == 0:
                metrics.append(0)
                continue
            cur_map = average_precision_score(y_true, y_score)
            # print("cur_map", cur_map)
            metrics.append(cur_map)
        except IndexError as e:
            metrics.append(0)
            continue
        p_idx += 1
        if p_idx % 20 == 0:
            print("map until now", np.mean(metrics), len(metrics), cur_map)
    
    print("average map", np.mean(metrics), len(metrics))


if __name__ == '__main__':
    find_paper_source_by_rule()
