from os.path import join
import os
import json
from tqdm import tqdm
import numpy as np
from fuzzywuzzy import fuzz
from cogdl import pipeline
from bs4 import BeautifulSoup
from sklearn.metrics import average_precision_score

import utils
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp



def extract_paper_citation_graph():
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = []
    dblp_fname = "DBLP-Citation-network-V15.1.json"
    parse_err_cnt = 0

    wf = open(join(data_dir, "dblp_pids.txt"), "w")
    with open(join(data_dir, dblp_fname), "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue
            if i % 10000 == 0: 
                logger.info("reading papers %d, parse err cnt %d", i, parse_err_cnt)
            try:
                paper_tmp = json.loads(line.strip())
                wf.write(paper_tmp["id"] + "\n")
                wf.flush()
            except:
                parse_err_cnt += 1
            papers.append(paper_tmp)
    wf.close()

    paper_dict_filter = {}
    for paper in tqdm(papers):
        paper_dict_filter[paper["id"]] = paper.get("references", [])

    logger.info("number of papers after filtering %d", len(paper_dict_filter))
    utils.dump_json(paper_dict_filter, data_dir, "dblp_papers_refs.json")


def merge_paper_references():
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    paper_dict_open = utils.load_json(data_dir, "dblp_papers_refs.json")
    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")

    for paper in tqdm(papers_train + papers_valid):
        pid = paper["_id"]
        cur_refs = paper.get("references", [])
        if len(cur_refs) == 0:
            continue
        refs_open = paper_dict_open.get(pid, [])
        refs_update = list(set(cur_refs + refs_open))
        paper_dict_open[pid] = refs_update
    
    utils.dump_json(paper_dict_open, data_dir, "dblp_papers_refs_merged.json")


def gen_paper_emb(year=2023, method="prone"):
    print("method", method)
    paper_dict = utils.load_json(settings.DATA_TRACE_DIR, "dblp_papers_refs_merged_{}.json".format(year))
    pids_set = set()
    edges = []
    for pid in tqdm(paper_dict):
        pids_set.add(pid)
        for ref_id in paper_dict[pid]:
            pids_set.add(ref_id)
            edges.append([pid, ref_id])
            edges.append([ref_id, pid])
    
    pids_sorted = sorted(list(pids_set))
    pid_to_idx = {pid: idx for idx, pid in enumerate(pids_sorted)}
    edges = [[pid_to_idx[pid], pid_to_idx[ref_id]] for pid, ref_id in edges]

    if method == "prone":
        generator = pipeline("generate-emb", model="prone")
    elif method == "line":
        generator = pipeline("generate-emb", model="line", walk_length=5, walk_num=5)
    elif method == "netsmf":
        generator = pipeline("generate-emb", model="netsmf", window_size=5, num_round=5)
    else:
        raise NotImplementedError

    # generate embedding by an unweighted graph
    edge_index = np.array(edges)
    print("genreate_emb...", edge_index.shape)
    outputs = generator(edge_index)
    print("outputs", outputs.shape)

    out_dir = join(settings.OUT_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, "paper_id.txt"), "w") as f:
        for pid in pids_sorted:
            f.write(pid + "\n")
            f.flush()
    np.savez(join(out_dir, "paper_emb_{}.npz".format(method)), emb=outputs)


def gen_paper_emb_kddcup(method="prone"):
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    print("method", method)
    paper_dict = utils.load_json(data_dir, "dblp_papers_refs_merged.json")
    pids_set = set()
    edges = []
    for pid in tqdm(paper_dict):
        pids_set.add(pid)
        for ref_id in paper_dict[pid]:
            pids_set.add(ref_id)
            edges.append([pid, ref_id])
            edges.append([ref_id, pid])

    
    pids_sorted = sorted(list(pids_set))
    pid_to_idx = {pid: idx for idx, pid in enumerate(pids_sorted)}
    edges = [[pid_to_idx[pid], pid_to_idx[ref_id]] for pid, ref_id in edges]

    if method == "prone":
        generator = pipeline("generate-emb", model="prone")
    elif method == "line":
        generator = pipeline("generate-emb", model="line", walk_length=5, walk_num=5)
    elif method == "netsmf":
        generator = pipeline("generate-emb", model="netsmf", window_size=5, num_round=5)
    else:
        raise NotImplementedError

    # generate embedding by an unweighted graph
    edge_index = np.array(edges)
    print("genreate_emb...", edge_index.shape)
    outputs = generator(edge_index)
    print("outputs", outputs.shape)

    out_dir = join(settings.OUT_DIR, "kddcup", method)
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, "paper_id.txt"), "w") as f:
        for pid in pids_sorted:
            f.write(pid + "\n")
            f.flush()
    np.savez(join(out_dir, "paper_emb_{}.npz".format(method)), emb=outputs)

    
def eval_node_sim(year=2023, method="prone"):
    data_year_dir = join(settings.DATA_TRACE_DIR, str(year))
    test_papers = utils.load_json(data_year_dir, "paper_source_trace_test.json")
    pids = []
    with open(join(settings.OUT_DIR, method, "paper_id.txt"), "r") as f:
        for line in f:
            pids.append(line.strip())
    pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
    emb = np.load(join(settings.OUT_DIR, method, "paper_emb_{}.npz".format(method)))["emb"]

    xml_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    metrics = []
    f_idx = 0
    for paper in tqdm(test_papers):
        pid = paper["_id"]
        file = join(xml_dir, pid + ".tei.xml")
        f = open(file, encoding='utf-8')

        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

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

        f.close()

        ref_id_to_score = {}
        ref_id_to_label = {}
        cur_emb = emb[pid_to_idx[pid]]
        cur_refs = paper.get("references", [])
        ref_truths = set([x["_id"] for x in paper.get("refs_trace", [])])
        for ref in cur_refs:
            ref_emb = emb[pid_to_idx[ref]]
            cur_sim = np.dot(cur_emb, ref_emb)
            cur_sim = 1/(1 + np.exp(-cur_sim))
            ref_id_to_score[ref] = cur_sim
            if ref in ref_truths:
                ref_id_to_label[ref] = 1
            else:
                ref_id_to_label[ref] = 0
        
        ref_id_to_score_sorted = sorted(ref_id_to_score.items(), key=lambda x: x[1], reverse=True)
        ref_labels = [ref_id_to_label[x[0]] for x in ref_id_to_score_sorted]
        truth_id_not_in = ref_truths - set(cur_refs)
        n_limit = n_refs - len(truth_id_not_in)
        scores = [x[1] for x in ref_id_to_score_sorted][:n_limit]
        labels = ref_labels[:n_limit]
        if len(truth_id_not_in) > 0:
            scores += [0] * len(truth_id_not_in)
            labels += [1] * len(truth_id_not_in)
        if len(scores) < n_refs:
            scores += [0] * (n_refs - len(scores))
            labels += [0] * (n_refs - len(labels))
        
        cur_map = average_precision_score(labels, scores)
        metrics.append(cur_map)
        f_idx += 1
        if f_idx % 20 == 0:
            print("map until now", np.mean(metrics), len(metrics), cur_map)
    
    print("prone map", np.mean(metrics), len(metrics))


def eval_node_sim_kddcup(method="prone", role="valid"):
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_{}_wo_ans.json".format(role))
    out_dir = join(settings.OUT_DIR, "kddcup", method)
    paper_info_more = utils.load_json(data_dir, "paper_info_hit_from_dblp.json")

    pids = []
    with open(join(out_dir, "paper_id.txt"), "r") as f:
        for line in f:
            pids.append(line.strip())
    pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
    emb = np.load(join(out_dir, "paper_emb_{}.npz".format(method)))["emb"]

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        ref_ids = paper.get("references", [])
        cur_title_to_pid = {}
        for ref_id in ref_ids:
            if ref_id in paper_info_more:
                cur_title_to_pid[paper_info_more[ref_id]["title"].lower()] = ref_id

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        cur_title_to_b_idx = {}
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
            cur_title_to_b_idx[ref.analytic.title.text.lower()] = b_idx - 1
            if b_idx > n_refs:
                n_refs = b_idx

        assert len(sub_example_dict[cur_pid]) == n_refs
        y_score = [0] * n_refs

        cur_emb = emb[pid_to_idx[cur_pid]]

        for r_idx, ref_id in enumerate(ref_ids):
            if ref_id not in paper_info_more:
                continue
            cur_title = paper_info_more[ref_id].get("title", "").lower()
            if len(cur_title) == 0:
                continue
            cur_b_idx = None
            for b_title in cur_title_to_b_idx:
                cur_sim = fuzz.ratio(cur_title, b_title)
                if cur_sim >= 80:
                    cur_b_idx = cur_title_to_b_idx[b_title]
                    break
            if cur_b_idx is None:
                continue
            ref_emb = emb[pid_to_idx[ref_id]]
            cur_sim = np.dot(cur_emb, ref_emb)
            cur_sim = utils.sigmoid(cur_sim)
            y_score[cur_b_idx] = float(cur_sim)

        print(y_score)
        sub_dict[cur_pid] = y_score
    
    utils.dump_json(sub_dict, out_dir, f"{role}_sub_{method}.json")


if __name__ == "__main__":
    method = "prone"
    extract_paper_citation_graph()
    merge_paper_references()
    # gen_paper_emb(method=method)
    gen_paper_emb_kddcup(method=method)
    # eval_node_sim(method=method)
    eval_node_sim_kddcup(method=method)
