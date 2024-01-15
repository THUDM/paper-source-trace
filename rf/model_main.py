import os
import numpy as np
import set_param
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.metrics import average_precision_score
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def calculate_TPFN(pre_result, label):
    TP, TN, FP, FN = 0, 0, 0, 0
    total = len(label)
    for item, jtem in zip(pre_result, label):
        if item == 1 and jtem == 1:
            TP += 1
        elif item == 0 and jtem == 1:
            FP += 1
        elif item == 1 and jtem == 0:
            FN += 1
        else:
            TN += 1
    print("Accuracy:", (TP + TN) / total)
    print("Precision:", TP / (TP + FP))
    print("Recall:", TP / (TP + FN))


# mode = "valid"
# mode = "train"
mode = "test"
# model_type = "SVM"
model_type = "RandomForest"
# model_type = "LR"


params = set_param.Args(model_type)
if mode == "train":
    data = np.loadtxt(open(f"processed_data/train_data.csv", "rb"), delimiter=",")
    label = np.loadtxt(open(f"processed_data/train_label.csv", "rb"), delimiter=",")
    if model_type == "SVM":
        model = svm.SVC(C=params.C, kernel=params.kernel, verbose=params.verbose, max_iter=params.max_iter,
                        tol=params.tol, probability=True)
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=params.n_estimators)
    elif model_type == "LR":
        model = LogisticRegression(solver=params.solver, multi_class=params.multi_class)
    model.fit(data, label)
    save_model = pickle.dumps(model)
    os.makedirs("saved_model", exist_ok=True)
    write_file = open(f"saved_model/{model_type}.pkl", 'wb')
    write_file.write(save_model)
    write_file.close()
    pre_result = model.predict(data)
    calculate_TPFN(pre_result, label)
elif mode == "test":
    with open('processed_data/test_data.json') as read_file:
        data_dic = json.load(read_file)
    with open('processed_data/test_label.json') as read_file:
        label_dic = json.load(read_file)
    with open(f'saved_model/{model_type}.pkl', 'rb') as read_file:
        model = pickle.load(read_file)
    
    map_list = []
    total_pre_list = []
    total_label = []
    for item in data_dic.keys():
        this_data = data_dic[item]
        this_label = [jtem[0] for jtem in label_dic[item]]
        pre_result = model.predict_proba(this_data)
        pre_result = [jtem[1] for jtem in pre_result]
        total_pre_list += [(1 if jtem >= 0.5 else 0) for jtem in pre_result]
        total_label += this_label
        map_list.append(average_precision_score(this_label, pre_result))
    calculate_TPFN(total_pre_list, total_label)
    print("MAP:", sum(map_list) / len(map_list))
    # 显示特征权重
    # feature_importances = model.feature_importances_
    # print(feature_importances)


elif mode == "valid":
    data = np.loadtxt(open(f"processed_data/valid/valid_data.csv", "rb"), delimiter=",")
    label = np.loadtxt(open(f"processed_data/valid/valid_label.csv", "rb"), delimiter=",")
    if model_type == "SVM":
        model = svm.SVC(C=params.C, kernel=params.kernel, verbose=params.verbose, max_iter=params.max_iter,
                        tol=params.tol, probability=True)
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=params.n_estimators)
    elif model_type == "LR":
        model = LogisticRegression(solver=params.solver, multi_class=params.multi_class)
    model.fit(data, label)
    predict = model.predict(data)
    calculate_TPFN(predict, label)
else:
    print("mode input error!")
    