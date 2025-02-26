import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import DataGenerator, get_word2index_dict


# 模型定义：CnnLstmModel
class CnnLstmModel(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9, conv_channels=[128], lstm_hidden_size=256, dropout=0.3,
                 bidirectional=True):
        super(CnnLstmModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)

        self.conv_layers = nn.ModuleList()
        in_channels = 128  # 输入通道数
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels

        # LSTM 层
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_size, batch_first=True,
                            bidirectional=bidirectional)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size * 2 if bidirectional else lstm_hidden_size, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Embedding 层
        out = self.embedding(x)  # [batch_size, 200, 128]

        # 卷积层
        out = out.permute(0, 2, 1)  # [batch_size, 128, 200]
        for conv_layer in self.conv_layers:
            out = conv_layer(out)

        out = out.permute(0, 2, 1)  # [batch_size, * , 128]

        # LSTM 层
        outputs, (h, c) = self.lstm(out)
        # out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)  # [batch_size, lstm_hidden_size * 2]
        if self.lstm.bidirectional:
            out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)  # [batch_size, lstm_hidden_size * 2]
        else:
            out = h[0, :, :]  # [batch_size, lstm_hidden_size]
        # 全连接层
        out = self.fc(out)

        return out


# 训练函数
def train_model(model, train_loader, val_loader, device, loss_func, optimizer, scheduler, num_epochs=30):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    min_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        y_true_train, y_pred_train = [], []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, preds = torch.max(output, 1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        # 训练集的评估指标
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted')
        train_f1 = f1_score(y_true_train, y_pred_train, average='weighted')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = loss_func(output, y)
                val_loss += loss.item() * x.size(0)
                _, preds = torch.max(output, 1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

                y_true_val.extend(y.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        # 验证集的评估指标
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted')
        val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        # 更新学习率
        scheduler.step()

        print(f"epoch:{epoch + 1}, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, "
              f"train_loss:{train_loss:.4f}, val_loss:{val_loss:.4f}")

        # 保存模型
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), f"newmodel/cnn_lstm_best.pth")

    return train_losses, val_losses, train_accs, val_accs, train_precisions, val_precisions, train_recalls, val_recalls, train_f1s, val_f1s


# 绘制结果
def plot_results(train_accs, train_losses, val_accs, val_losses):
    plt.figure(figsize=(12, 6))

    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, "r", label="train")
    plt.plot(val_accs, "g", label="val")
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, "r", label="train")
    plt.plot(val_losses, "g", label="val")
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('newmodel/cnn_lstm_validation_results.png')
    plt.show()


# 测试集评估函数
def evaluate_test(model, test_loader, device):
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    y_true_test, y_pred_test = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, preds = torch.max(output, 1)
            loss = nn.CrossEntropyLoss()(output, y)
            test_loss += loss.item() * x.size(0)
            test_correct += (preds == y).sum().item()
            test_total += x.size(0)

            y_true_test.extend(y.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    # 计算指标
    test_loss /= test_total
    test_acc = test_correct / test_total
    test_precision = precision_score(y_true_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_true_test, y_pred_test, average='weighted')
    test_f1 = f1_score(y_true_test, y_pred_test, average='weighted')

    print(f"Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    return test_acc, test_precision, test_recall, test_f1


# 验证实验
def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 datasets 中的 get_word2index_dict
    word2index_dict = get_word2index_dict(n_common=5000)
    num_classes = 12

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        DataGenerator(root="datasets/data_train.txt", word2index_dict=word2index_dict, max_len=300),
        batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        DataGenerator(root="datasets/data_val.txt", word2index_dict=word2index_dict, max_len=300),
        batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        DataGenerator(root="datasets/data_test.txt", word2index_dict=word2index_dict, max_len=300),
        batch_size=16, shuffle=False)

    # 不同参数配置
    configurations = [
        # dropout
        {"conv_channels": [128, 256], "lstm_hidden_size": 256, "dropout": 0, "bidirectional": True},
        {"conv_channels": [128, 256], "lstm_hidden_size": 256, "dropout": 0.3, "bidirectional": True},
        {"conv_channels": [128, 256], "lstm_hidden_size": 256, "dropout": 0.5, "bidirectional": True},
        {"conv_channels": [128, 256], "lstm_hidden_size": 256, "dropout": 0.7, "bidirectional": True},

        # bidirectional
        {"conv_channels": [128, 256], "lstm_hidden_size": 256, "dropout": 0.5, "bidirectional": False},

        # lstm_hidden_size
        {"conv_channels": [128, 256], "lstm_hidden_size": 128, "dropout": 0.5, "bidirectional": True},
        {"conv_channels": [128, 256], "lstm_hidden_size": 512, "dropout": 0.5, "bidirectional": True},

        # conv_channels
        {"conv_channels": [64, 128], "lstm_hidden_size": 256, "dropout": 0.5, "bidirectional": True},
        {"conv_channels": [256, 512], "lstm_hidden_size": 256, "dropout": 0.5, "bidirectional": True},
    ]

    for config in configurations:
        print(f"Testing Configuration: {config}")

        # 初始化模型
        model = CnnLstmModel(
            num_vocabulary=len(word2index_dict), num_classes=num_classes,
            conv_channels=config["conv_channels"], lstm_hidden_size=config["lstm_hidden_size"],
            dropout=config["dropout"], bidirectional=config["bidirectional"]
        ).to(device)

        # 损失函数与优化器
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        # 训练模型
        train_losses, val_losses, train_accs, val_accs, train_precisions, val_precisions, train_recalls, val_recalls, train_f1s, val_f1s = train_model(
            model, train_loader, val_loader, device, loss_func, optimizer, scheduler, num_epochs=40)

        # 绘制实验结果
        plot_results(train_accs, train_losses, val_accs, val_losses)

        # 输出最终指标
        print(f"Configuration: {config}")
        '''
        print(f"Train Accuracy: {train_accs[-1]:.4f}, Train Precision: {train_precisions[-1]:.4f}, "
              f"Train Recall: {train_recalls[-1]:.4f}, Train F1: {train_f1s[-1]:.4f}")
        print(f"Validation Accuracy: {val_accs[-1]:.4f}, Validation Precision: {val_precisions[-1]:.4f}, "
              f"Validation Recall: {val_recalls[-1]:.4f}, Validation F1: {val_f1s[-1]:.4f}")
        '''

        # 在测试集上评估模型
        evaluate_test(model, test_loader, device)


if __name__ == "__main__":
    validate()
