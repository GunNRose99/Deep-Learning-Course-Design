import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import BiLstmModel, CnnModel, CnnLstmModel
from torch.utils.data import DataLoader
from datasets import DataGenerator, get_word2index_dict, get_random_word2index_dict


def train_cnn_lstm(model_name):
    device = torch.device("cpu")
    word2index_dict = get_word2index_dict(n_common=5000)
    # word2index_dict = get_random_word2index_dict(n_common=8000)
    if model_name == "lstm":
        model = BiLstmModel(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    elif model_name == "cnn":
        model = CnnModel(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    elif model_name == "cnn-lstm":
        model = CnnLstmModel(num_vocabulary=len(word2index_dict), num_classes=12).to(device)
    else:
        raise KeyError("ERROR!")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_func = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        DataGenerator(root="datasets/data_train.txt", word2index_dict=word2index_dict, max_len=300), batch_size=16,
        shuffle=True)
    val_loader = DataLoader(DataGenerator(root="datasets/data_val.txt", word2index_dict=word2index_dict, max_len=300),
                            batch_size=16, shuffle=False)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    min_loss = 10000
    for epoch in range(30):
        data_train = tqdm(train_loader)
        losses = []
        labels_true, labels_pred = np.array([]), np.array([])
        model.train()
        for batch, (x, y) in enumerate(data_train):
            labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)
            input_ids, labels = x.to(device), y.to(device)
            y_prob = model(input_ids)
            y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, y_pred], axis=-1)
            loss = loss_func(y_prob, labels)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_train.set_description_str(
                f"epoch:{epoch + 1},batch:{batch + 1},loss:{loss.item():.5f},lr:{scheduler.get_last_lr()[0]:.7f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/{model_name}_epoch{epoch + 1}.pth")

        train_acc = np.mean(labels_true == labels_pred)
        train_loss = np.mean(losses)
        scheduler.step()

        data_val = tqdm(val_loader)
        losses = []
        labels_true, labels_pred = np.array([]), np.array([])
        model.eval()
        with torch.no_grad():
            for x, y in data_val:
                labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)
                input_ids, labels = x.to(device), y.to(device)
                y_prob = model(input_ids)
                y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
                labels_pred = np.concatenate([labels_pred, y_pred], axis=-1)
                loss = loss_func(y_prob, labels)
                losses.append(loss.item())

        val_acc = np.mean(labels_true == labels_pred)
        val_loss = np.mean(losses)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")

        print(f"epoch:{epoch + 1},train_acc:{train_acc:.4f},val_acc:{val_acc:.4f},train_loss:{train_loss:.4f},val_loss:{val_loss:.4f}")

    plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name=model_name)


def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accs) + 1), train_accs, "r", label="train")
    plt.plot(range(1, len(val_accs) + 1), val_accs, "g", label="val")
    plt.title(f"{model_name}_accuracy-epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, "r", label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, "g", label="val")
    plt.title(f"{model_name}_loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"images/{model_name}_epoch_acc_loss.jpg")


if __name__ == '__main__':
    train_cnn_lstm(model_name="lstm")
    train_cnn_lstm(model_name="cnn")
    train_cnn_lstm(model_name="cnn-lstm")