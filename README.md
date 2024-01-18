# paper-source-trace

## Prerequisites
- Linux
- Python 3.9
- PyTorch 1.10.0+cu111

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/paper-source-trace.git
cd paper-source-trace
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

## Dataset
The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1I_HZXBx7U0UsRHJL5JJagw?pwd=bft3) with password bft3 or [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/paper-source-trace/paper-source-trace-data.zip).
Please put the _data_ folder and _out_ folder into the project directory.
The paper XML files are generated by [Grobid](https://grobid.readthedocs.io/en/latest/Introduction/) APIs from paper pdfs.

## How to Run
```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used

python rule.py  # rule-based method
python bert.py  # BERT-based method
python net_emb.py  # network embedding based method (ProNE)

python glm/test_glm.py  # test glm
bash glm/run_finetune_ds_10b.sh  # train glm-10b model
bash glm/run_finetune_ds.sh  # train glm-2b model

python chatglm/data/positive_negetive_balance.py  # make the positive data and negative data in dataset balance(1:1)
python chatglm/data/map.py  # calculate map
bash chatglm/test.sh  # test chatglm model(pt-v2)
bash chatglm/ds_test_finetune.sh  # test chatglm model(finetune)

python gpt-api/gpt.py  # use gpt model.Token and api-base are needed.
python gpt-api/map.py  # calculate map

python claude/claude.py  #use claude model.URL and key are needed.

python galactica/dataset.py  #make dataset
python galactica/main.py  # finetune galactica
```

## Results
Evaluation metrics: average MAP

|       | MAP   |
|-------|-------|
| Rule  | 0.0565 |
| ProNE | 0.1289 |
| BERT  | 0.1294 |
