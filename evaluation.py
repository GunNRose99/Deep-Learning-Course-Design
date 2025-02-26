import torch
import seaborn
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import DataGenerator, get_word2index_dict
from layers import CnnLstmModel_1,CnnLstmModel_2,CnnLstmModel_3,CnnLstmModel_4
from nets import BiLstmModel, CnnLstmModel, CnnModel
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, precision_score, f1_score, \
    recall_score, auc, roc_curve

rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
seaborn.set(context='notebook', style='ticks', rc=rc)


def onehot(y, num_classes=12):
    y_ = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        y_[i, int(y[i])] = 1

    return y_


def softmax(y):
    y_exp = np.exp(y)
    for i in range(len(y)):
        y_exp[i, :] = y_exp[i, :] / np.sum(y_exp[i, :])

    return y_exp


def get_test_result(model_path, columns, root, model_name):
    print(model_name)
    device = torch.device("cpu")
    word2index_dict = get_word2index_dict(5000)
    if model_name == "CnnLstmModel_1":
        model = CnnLstmModel_1(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    elif model_name == "CnnLstmModel_2":
        model = CnnLstmModel_2(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    elif model_name == "CnnLstmModel_3":
        model = CnnLstmModel_3(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    elif model_name == "CnnLstmModel_4":
        model = CnnLstmModel_4(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    else:
        raise KeyError("ERROR!")

    '''
    if model_name == "lstm":
        model = BiLstmModel(num_vocabulary=len(word2index_dict), num_classes=12)
    elif model_name == "cnn":
        model = CnnModel(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    elif model_name == "cnn-lstm":
        model = CnnLstmModel(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    else:
        raise KeyError("model_name error!")
    '''

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(DataGenerator(root=root, word2index_dict=word2index_dict, max_len=300), batch_size=32,
                             shuffle=False)
    data = tqdm(test_loader)
    labels_true, labels_pred, labels_prob = np.array([]), np.array([]), []
    with torch.no_grad():
        for x, y in data:
            datasets_test = x.to(device)
            prob = model(datasets_test)
            labels_prob.append(prob.cpu().numpy())
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, pred], axis=-1)
            labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)

    labels_prob = nn.Softmax(dim=-1)(torch.FloatTensor(np.concatenate(labels_prob, axis=0))).numpy()
    accuracy = accuracy_score(labels_true, labels_pred)
    precision = precision_score(labels_true, labels_pred, average="weighted")
    recall = recall_score(labels_true, labels_pred, average="weighted")
    f1 = f1_score(labels_true, labels_pred, average="weighted")
    print(f"{model_name},accuracy:{accuracy:.3f},precision:{precision:.3f},recall:{recall:.3f},f1:{f1:.3f}")
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "r--")
    for i in range(12):
        fpr, tpr, _ = roc_curve(onehot(labels_true,num_classes=12)[:, i], labels_prob[:, i])
        plt.plot(fpr, tpr, label=f"{columns[i]} AUC:{auc(fpr, tpr):.3f}", linewidth=4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("roc_curve")
    plt.legend()
    plt.savefig(f"layers_image/{model_name}_roc_curve.jpg")
    plt.figure(figsize=(10, 10))
    for i in range(12):
        p, r, thresholds = precision_recall_curve(onehot(labels_true,num_classes=12)[:, i], labels_prob[:, i])
        plt.plot(r, p, label=columns[i], linewidth=4)
    plt.title("precision_recall_Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f"layers_image/{model_name}_pr_curve.jpg")
    matrix = pd.DataFrame(confusion_matrix(labels_true, labels_pred, normalize="true"), columns=columns, index=columns)
    plt.figure(figsize=(10, 10))
    seaborn.heatmap(matrix, annot=True, cmap="GnBu")
    plt.title("confusion_matrix")
    plt.savefig(f"layers_image/{model_name}_confusion_matrix.jpg")


if __name__ == '__main__':
    get_test_result(model_path="layers4/CnnLstmModel_1_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="CnnLstmModel_1")
    get_test_result(model_path="layers4/CnnLstmModel_2_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="CnnLstmModel_2")
    get_test_result(model_path="layers4/CnnLstmModel_3_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="CnnLstmModel_3")
    get_test_result(model_path="layers4/CnnLstmModel_4_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="CnnLstmModel_4")
    '''
    get_test_result(model_path="models/lstm_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="lstm")
    get_test_result(model_path="models/cnn_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="cnn")
    get_test_result(model_path="models/cnn-lstm_best.pth", columns=['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人'], root="datasets/data_test.txt", model_name="cnn-lstm")
    '''





