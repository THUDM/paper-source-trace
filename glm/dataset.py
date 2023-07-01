import os
import json
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.variable_num_choices = True
        self.example_list = []
        with open(path, "r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                item = json.loads(line)
                item["idx"] = str(idx)
                item["answer"]= item["choices_pretokenized"][item["label"]]
                self.example_list.append(item)
        # self.example_list = self.example_list[:200]  # debug
        self.examples = {example["idx"]: example for example in self.example_list}
        print(f"Creating {len(self.example_list)} examples")
        self.dataset_name = "multichoice-" + os.path.basename(path).split(".")[0]

        contexts = [x["inputs_pretokenized"][-500:] for x in self.example_list]
        candidates = [x["choices_pretokenized"] for x in self.example_list]
        self.labels = [x["label"] for x in self.example_list]
        answers = [x["answer"] for x in self.example_list]
        self.input_dict = {"contexts": contexts, "candidates": candidates, "labels": self.labels, "answers": answers}


    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.input_dict.items()}
    