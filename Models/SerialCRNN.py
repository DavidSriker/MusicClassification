import torch.nn as nn


NUM_CONV_LAYERS = 3
KERNEL_SIZE = 5
POOLING_SIZE = 2
DROPOUT_PROB = 0.2
CONV_FILTER_COUNT = 64
BATCH_SIZE = 32
LSTM_HIDDEN = 96
DENSE_SIZE = 48
NUM_CLASSES = 8

EPOCH_COUNT = 70
NUM_HIDDEN = 64
L2_regularization = 0.001

class SerialCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=CONV_FILTER_COUNT):
        super(SerialCNN, self).__init__()
        model = []
        for _ in range(NUM_CONV_LAYERS):
            model += [nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=KERNEL_SIZE),
                      nn.BatchNorm2d(num_features=out_channels),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(POOLING_SIZE),
                      nn.Dropout(DROPOUT_PROB)]
            in_channels, out_channels = out_channels, out_channels * 2
        self.conv_model = nn.Sequential(*model)
        self.size = in_channels

        model = []
        model += [nn.Linear(912, in_channels), nn.Dropout(DROPOUT_PROB)]  # 912 because H, W = 76, 12
        model += [nn.LSTM(input_size=in_channels, hidden_size=LSTM_HIDDEN, num_layers=2, bidirectional=True)]
        self.lstm_block = nn.Sequential(*model)
        # dense layers
        self.linear = nn.Sequential(*[nn.Linear(LSTM_HIDDEN * 2, 1),
                                      nn.Dropout(DROPOUT_PROB)])

        model = []
        model += [nn.Linear(self.size, DENSE_SIZE)]

        # softmax layer
        model += [nn.Linear(DENSE_SIZE, NUM_CLASSES),
                  nn.Softmax(dim=1)]

        self.out_block = nn.Sequential(*model)

    def forward(self, x):
        conv_res = self.conv_model(x)
        out, hid = self.lstm_block(conv_res.reshape((*conv_res.shape[0:2], conv_res.shape[2] * conv_res.shape[3])))
        out = self.linear(out).squeeze()
        return self.out_block(out)
