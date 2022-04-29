import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

cfg = utils.read_yaml()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=cfg['model']['conv1_in_channel'],
                               out_channels=cfg['model']['conv1_out_channel'],
                               kernel_size=cfg['model']['conv1_kernel_size'])
        self.conv2 = nn.Conv2d(in_channels=cfg['model']['conv2_in_channel'],
                               out_channels=cfg['model']['conv2_out_channel'],
                               kernel_size=cfg['model']['conv2_kernel_size'],
                               stride=cfg['model']['conv2_stride'])
        self.conv3 = nn.Conv2d(in_channels=cfg['model']['conv3_in_channel'],
                               out_channels=cfg['model']['conv3_out_channel'],
                               kernel_size=cfg['model']['conv3_kernel_size'],
                               stride=cfg['model']['conv3_stride'])
        self.conv4 = nn.Conv2d(in_channels=cfg['model']['conv4_in_channel'],
                               out_channels=cfg['model']['conv4_out_channel'],
                               kernel_size=cfg['model']['conv4_kernel_size'],
                               stride=cfg['model']['conv4_stride'])

        self.dropout1 = nn.Dropout(cfg['model']['dropout_1'])
        self.dropout2 = nn.Dropout(cfg['model']['dropout_2'])

        self.batchnorm1 = nn.BatchNorm2d(cfg['model']['conv1_out_channel'])
        self.batchnorm2 = nn.BatchNorm2d(cfg['model']['conv2_out_channel'])
        self.batchnorm3 = nn.BatchNorm2d(cfg['model']['conv3_out_channel'])
        self.batchnorm4 = nn.BatchNorm2d(cfg['model']['conv4_out_channel'])

        self.fc1 = nn.Linear(in_features=cfg['model']['fc1_in_features'],
                             out_features=cfg['model']['fc1_out_features'])
        self.fc2 = nn.Linear(in_features=cfg['model']['fc2_in_features'],
                             out_features=cfg['model']['fc2_out_features'])

    def forward(self, features):
        x = self.conv1(features)
        x = F.relu(self.batchnorm1(x))
        x = F.max_pool2d(x, cfg['model']['max_pool'])
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        #x = F.max_pool2d(x, cfg['model']['max_pool'])
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))
        #x = F.max_pool2d(x, cfg['model']['max_pool'])
        x = self.dropout1(x)
        
        
        #x = self.conv4(x)
        #x = F.relu(self.batchnorm4(x))
        #x = F.max_pool2d(x, cfg['model']['max_pool'])
        #x = self.dropout1(x)
        
    
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        out = self.fc2(x)
        #out = F.softmax(out, dim=1)
        return out


class RnnModel(nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()
        # Bidirectional LSTM
        self.biLSTM = nn.LSTM(input_size=cfg['model']['rnn_input_size'],
                                hidden_size=cfg['model']['rnn_hidden_size'],
                                num_layers=cfg['model']['rnn_num_layers'],
                                batch_first=True,
                                bidirectional=True)
        

        self.lstm = nn.LSTM(input_size=cfg['model']['rnn_hidden_size'],
                                hidden_size=cfg['model']['rnn_hidden_size_2'],
                                num_layers=cfg['model']['rnn_num_layers'],
                                batch_first=True,
                                bidirectional=False)
        
        self.fc = nn.Linear(in_features=cfg['model']['rnn_hidden_size_2'],
                            out_features=cfg['model']['fc2_out_features'])
    
    def forward(self, x):

        x, _ = self.biLSTM(x)
        x = F.relu(x)
        
        x, _ = self.lstm(x)
        x = F.relu(x) 
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
