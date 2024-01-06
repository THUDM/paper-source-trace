from os.path import join
import os
from tqdm import tqdm
import numpy as np
from cogdl import pipeline
from bs4 import BeautifulSoup
from sklearn.metrics import average_precision_score

import utils
import settings


def gen_paper_emb(year=2023):
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

    generator = pipeline("generate-emb", model="prone")

    # generate embedding by an unweighted graph
    edge_index = np.array(edges)
    print("genreate_emb...", edge_index.shape)
    outputs = generator(edge_index)
    print("outputs", outputs.shape)

    out_dir = join(settings.OUT_DIR, "prone")
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, "paper_id.txt"), "w") as f:
        for pid in pids_sorted:
            f.write(pid + "\n")
            f.flush()
    np.savez(join(out_dir, "paper_emb.npz"), emb=outputs)


def eval_node_sim(year=2023):
    data_year_dir = join(settings.DATA_TRACE_DIR, str(year))
    test_papers = utils.load_json(data_year_dir, "paper_source_trace_test.json")
    pids = []
    with open(join(settings.OUT_DIR, "prone", "paper_id.txt"), "r") as f:
        for line in f:
            pids.append(line.strip())
    pid_to_idx = {pid: idx for idx, pid in enumerate(pids)}
    emb = np.load(join(settings.OUT_DIR, "prone", "paper_emb.npz"))["emb"]

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


if __name__ == "__main__":
    gen_paper_emb()
    eval_node_sim()
