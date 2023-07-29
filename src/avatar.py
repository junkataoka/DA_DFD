import torch.nn as nn
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

class WAVATAR(nn.Module):
    def __init__(self, C_in, class_num):
        super(WAVATAR, self).__init__()
        # wdcnn
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=16, kernel_size=64, stride=16, padding=1),
            nn.BatchNorm1d(16, affine=True),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),  # 默认padding=0

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=2816, out_features=2816//2),
            nn.ReLU(),
            nn.Linear(in_features=2816//2, out_features=2816//4),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2816//4, out_features=class_num+1),
            nn.Softmax(dim=1)
        )

        # self.domain_classifier = nn.Sequential()
        # self.domain_classifier.add_module('d_fc1', nn.Linear(in_features=100, out_features=100))
        # self.domain_classifier.add_module('d-l-relu', nn.LeakyReLU())
        # self.domain_classifier.add_module('d_classifier', nn.Linear(in_features=100, out_features=2))
        # self.domain_classifier.add_module('d-softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        # print("x shape: ", x.shape)
        x = self.net(x)
        # print("x shape: ", x.shape)
        x = x.view(x.size()[0], -1)
        #print("x shape: ", x.shape)
        x = self.fc(x)
        class_output = self.classifier(x)

        prob_p_dis = class_output[:, -1].unsqueeze(1)
        prob_p_class = class_output[:, :-1]
        prob_p_class = prob_p_class / (1-prob_p_dis+1e-6)

        return prob_p_class, prob_p_dis, x