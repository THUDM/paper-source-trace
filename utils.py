from os.path import join
import json
import numpy as np
import pickle
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from datetime import datetime

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


def find_bib_context(xml, dist=100):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = dd(list)
    bibr_strs_to_bid_id = {}
    for item in bs.find_all(type='bibr'):
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]
        item_str = "<ref type=\"bibr\" target=\"{}\">{}</ref>".format(item.attrs["target"], item.get_text())
        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [ii for ii in range(len(xml)) if xml.startswith(item_str, ii)]
        for pos in cur_bib_context_pos_start:
            bib_to_context[bib_id].append(xml[pos - dist: pos + dist].replace("\n", " ").replace("\r", " ").strip())
    return bib_to_context


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Log:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(file_path, 'w+')

    def log(self, s):
        self.f.write(str(datetime.now()) + "\t" + s + '\n')
        self.f.flush()