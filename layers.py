import torch
import torch.nn as nn

# 1. 1 层 LSTM + 1 层 CNN
class CnnLstmModel_1(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9):
        super(CnnLstmModel_1, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)
        # 1 层 CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )
        # 1 层 LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=True)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = out.permute(0, 2, 1)
        outputs, (h, c) = self.lstm(out)
        out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
        out = self.fc(out)
        return out


# 2. 1 层 LSTM + 2 层 CNN
class CnnLstmModel_2(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9):
        super(CnnLstmModel_2, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)

        # 2 层 CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )

        # 1 层 LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        outputs, (h, c) = self.lstm(out)
        out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
        out = self.fc(out)
        return out


# 3. 2 层 LSTM + 1 层 CNN
class CnnLstmModel_3(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9):
        super(CnnLstmModel_3, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)

        # 1 层 CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )

        # 2 层 LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = out.permute(0, 2, 1)
        outputs, (h, c) = self.lstm(out)
        out = torch.cat([h[-1, :, :], h[-2, :, :]], dim=-1)
        out = self.fc(out)
        return out


# 4. 2 层 LSTM + 2 层 CNN
class CnnLstmModel_4(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9):
        super(CnnLstmModel_4, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)

        # 2 层 CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )

        # 2 层 LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        outputs, (h, c) = self.lstm(out)
        out = torch.cat([h[-1, :, :], h[-2, :, :]], dim=-1)
        out = self.fc(out)
        return out
