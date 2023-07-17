import json
import os
import csv
import pymongo
from bson import ObjectId
import random
from lxml import etree
from fuzzywuzzy import fuzz
import re
from tqdm import tqdm

random.seed(1)  # 随机数种子，用来确定训练和验证集中用抽中哪些论文。如果改动，需要重新从数据库中查询数据。（查询语句留着，但是地址，用户名和密码删掉了，需要的话再填一下）


class Process_data(object):
    def __init__(self, paper_dic, mode):
        self.paper_dic = paper_dic
        paper_id = paper_dic['_id']
        paper_positive_id = [item['_id'] for item in paper_dic['refs_trace']]
        self.authors_set = set([item.get('name') for item in paper_dic.get('authors', {})])
        # 通过xml获取tree和listBibl
        try:
            path = f'paper-source-trace/paper-xml/{paper_id}.tei.xml'
            self.tree = etree.parse(path)
            root = self.tree.getroot()
            listBibl = root.xpath("//*[local-name()='listBibl']")[0]
            self.biblStruct = listBibl.getchildren()
            self.num_ref = len(self.biblStruct)
        except OSError:
            self.tree = None
            self.num_ref = 0
            print('not exits xml ' + paper_id)
        # 获取论文引用数
        self.reference_num = self.get_reciprocal_of_reference_num()
        if mode == 'test':  # train和valid需要随机选择部分论文构造正例，而test则可以直接把所有例子都放入query_list
            query_list = paper_dic.get('references', [])
        else:
            references = paper_dic.get('references', [])
            for item in paper_positive_id:
                try:
                    references.remove(item)
                except ValueError:
                    continue
            query_list = random.sample(references, min(max(len(paper_positive_id), 1),
                                                       len(references)))  # 最少选一个负例,最多references个（存在正例数多于reference的情况！）
            query_list += paper_positive_id
        self.data_list = []
        self.label_list = []
        for i, item in enumerate(query_list):
            this_data = []
            # query = {'_id': ObjectId(item)}
            # self.query_result = list(collection.find(query))
            self.query_result = saved_data.get(item, {})
            reference_place_list = self.get_referenced_place_num(item)
            if len(reference_place_list) == 0:
                continue  # 如果返回长度为0说明文章标题在xml的reference中没有找到自己的序号，这里暂时先不管它。
            this_data.append(self.get_referenced_num())
            this_data.append(self.get_common_authors(item))
            this_data.append(self.reference_num)
            this_data.append(self.key_words())
            this_data += reference_place_list
            self.data_list.append(this_data)
            self.label_list.append([1] if item in paper_positive_id else [0])
    
    # ONE 被引用次数
    def get_referenced_num(self):
        return self.query_result.get('n_citation', 0)
    
    # TWO,SIX,EIGHT 引用位置, 是否出现在图表中, 引用次数/引用总数
    # 0 abstract
    # 1 introduction
    # 2 related work
    # 3 method
    # 4 graph and figure
    # 5 result
    # 6 others
    def get_referenced_place_num(self, paper_id):
        # 从数据库中检索到title
        # query = {"_id": ObjectId(paper_id)}
        # title = list(collection.find(query))[0]['title']
        title = self.query_result.get('title', '')
        # 从xml中检索到序号
        if self.tree is None:
            return [0 * 8]
        
        paper_number = -1
        for i, item in enumerate(self.biblStruct):
            this_test = item.xpath('.//*[local-name()="title"]')
            this_text = this_test[0].text
            if this_text is None:
                try:
                    this_text = this_test[1].text
                except IndexError:
                    this_text = ''
            try:
                score = fuzz.partial_ratio(title, this_text)
            except ValueError:
                score = 0
            if score >= 80:
                paper_number = i + 1
                break
        place_num = [0 for i in range(8)]
        self.paper_number = paper_number
        if paper_number == -1:
            return place_num
        # 使用序号，在xml文件中检索位置
        nodes = self.tree.xpath(f"//*[contains(text(), '[{paper_number}]')]")
        reference_times = len(nodes)
        
        for item in nodes:
            found_text = ''
            this_node = item
            while found_text == '':
                this_node = this_node.getparent()
                if this_node is None:
                    break
                if this_node.xpath("local-name()") == 'figure':
                    place_num[4] = 1
                it_children = this_node.iterchildren()
                for jtem in it_children:
                    node = this_node
                    if jtem.xpath("local-name()") == 'head':
                        found_text = node.text
                        n_num = jtem.attrib.get('n')
                        node = this_node
                        if n_num is None:
                            break
                        while not n_num.isdigit():
                            node = node.getprevious()
                            if node is None:
                                break
                            node_children = node.iterchildren()
                            for ktem in node_children:
                                if ktem.xpath("local-name()") == 'head':
                                    n = ktem.attrib.get('n')
                                    if n is not None and n.isdigit():
                                        n_num = ktem.attrib.get('n')
                                        found_text = ktem.text
                                        break
                                    break
            
            if this_node is None or found_text == '':
                place_num[6] = 1
                continue
            if found_text is not None:
                found_text = found_text.lower()
            if fuzz.partial_ratio('abstract', found_text) >= 60:
                place_num[0] = 1
            elif fuzz.partial_ratio('introduction', found_text) >= 60:
                place_num[1] = 1
            elif fuzz.partial_ratio('related work', found_text) >= 60:
                place_num[2] = 1
            elif fuzz.partial_ratio('method', found_text) >= 60:
                place_num[3] = 1
            elif fuzz.partial_ratio('result', found_text) >= 60 or fuzz.partial_ratio('experiment', found_text) >= 60:
                place_num[5] = 1
            else:
                place_num[6] = 1
        pattern = re.compile(r'[\d+]')
        nodes = self.tree.xpath("//*[re:match(text(), $pattern)]",
                                namespaces={"re": "http://exslt.org/regular-expressions"},
                                pattern=pattern.pattern)
        total_ref_num = len(nodes)
        if not total_ref_num == 0:
            place_num[7] = reference_times / total_ref_num
        return place_num
    
    # FOUR 重叠作者
    def get_common_authors(self, paper_id):
        # ref_authors_set = set([item.get('name') for item in self.query_result.get('authors', {})])
        ref_authors_set = set(self.query_result.get('authors', []))
        if not len(self.authors_set & ref_authors_set) == 0:
            return 1
        else:
            return 0
    
    # FIVE 关键词
    def key_words(self):
        if self.paper_number == -1:
            return 0
        pattern = re.compile(r'[\d+]')
        nodes = self.tree.xpath("//*[re:match(text(), $pattern)]",
                                namespaces={"re": "http://exslt.org/regular-expressions"},
                                pattern=pattern.pattern)
        key_words_list = ['motivated by', 'inspired by']
        for item in nodes:
            if item.xpath('local-name()') == 'ref':
                node_text = item.getparent().text
            else:
                node_text = item.text
            if node_text is None:
                return 0
            node_text = node_text.lower()
            for jtem in key_words_list:
                pattern = re.compile(fr"{jtem}")
                match = pattern.search(node_text)
                if match is not None:
                    return 1
        return 0
    
    # SEVEN
    
    def get_reciprocal_of_reference_num(self):
        if self.num_ref == 0:
            return 0
        else:
            return 1 / self.num_ref


# mode = 'valid'
# mode = 'train'
mode = 'test'
with open(f"paper-source-trace/paper_source_trace_{mode}.json", 'r', encoding='utf-8') as read_file:
    data_dic = json.load(read_file)
all_id = [item['_id'] for item in data_dic]
data_list = []
label_list = []

# 连接数据库
# client = pymongo.MongoClient(host='', port=-1, authSource='', username='', password='')
# db = client.get_database('')
# collection = db[""]
with open('processed_data/saved_data.json', 'r') as read_file:
    saved_data = json.load(read_file)
n = 0
# 调用方法获取数据，然后将数据写入csv或json文件
if mode == 'test':
    total_data_dic = {}
    total_label_dic = {}
    for i, item in tqdm(enumerate(all_id), total=len(all_id)):
        process_data = Process_data(data_dic[i], mode)
        this_data, this_label = process_data.data_list, process_data.label_list
        total_data_dic[item] = this_data
        total_label_dic[item] = this_label
    with open(f'processed_data/{mode}/{mode}_data.json', 'w') as write_file:
        write_file.write(json.dumps(total_data_dic))
    with open(f'processed_data/{mode}/{mode}_label.json', 'w') as write_file:
        write_file.write(json.dumps(total_label_dic))
else:
    # train/valid:
    for i, item in tqdm(enumerate(all_id), total=len(all_id)):
        process_data = Process_data(data_dic[i], mode)
        this_data, this_label = process_data.data_list, process_data.label_list
        data_list += this_data
        label_list += this_label
    with open(f'processed_data/{mode}/{mode}_label.csv', 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(label_list)
    with open(f'processed_data/{mode}/{mode}_data.csv', 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(data_list)
