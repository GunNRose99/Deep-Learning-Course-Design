import torch
import torch.nn as nn


class BiLstmModel(nn.Module):

    def __init__(self, num_vocabulary, num_classes=2):
        super(BiLstmModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        outputs, (h, c) = self.lstm(out)
        out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
        out = self.dropout(out)
        out = self.linear(out)

        return out


class CnnModel(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9):
        super(CnnModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512 * 23, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class CnnLstmModel(nn.Module):
    def __init__(self, num_vocabulary, num_classes=9):
        super(CnnLstmModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_vocabulary, embedding_dim=128)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.5)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=True)

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


