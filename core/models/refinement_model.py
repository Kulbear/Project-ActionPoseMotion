"""
V1 is a sequential model where V2 is a residual learning version of V1.
"""
import torch.nn as nn


class RefineNetV1(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5, seq_len=45):
        super(RefineNetV1, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.hid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hid_dim // 2, self.hid_dim)
        )

        self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x)
        x = self.post_rnn(x)
        return {'refined_3d': x}


class RefineNetV2(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5, seq_len=45):
        super(RefineNetV2, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.hid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hid_dim // 2, self.hid_dim)
        )

        self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        identity = x
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x)
        x = self.post_rnn(x)
        x = x + identity
        return {'refined_3d': x}


class RefineNetV3(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5, seq_len=45):
        super(RefineNetV3, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.hid_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.seq_len),
            nn.Linear(self.hid_dim // 2, self.hid_dim)
        )

        self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x)
        x = self.post_rnn(x)
        return {'refined_3d': x}


class RefineNetV4(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5, seq_len=45):
        super(RefineNetV4, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.hid_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.seq_len),
            nn.Linear(self.hid_dim // 2, self.hid_dim)
        )

        self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        identity = x
        # print(x.size())
        x = self.pre_rnn(x)
        # print(x.size())
        x, hidden = self.rnn(x)
        x = self.post_rnn(x)
        x = x + identity
        return {'refined_3d': x}


class RefineConvNet(nn.Module):
    def __init__(self, *args, size=(64, 28, 45), **kwargs):
        super(RefineConvNet, self).__init__()

        self.size = size
        self.seq_len = size[1]
        self.n_joint = size[2] // 3

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 3), stride=(3, 1),
                      padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(3, 1),
                      padding=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.latent = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(3, 1),
                               padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=(3, 3), stride=(3, 1),
                               padding=1)
        )

    def forward(self, x):
        size = x.size()
        if len(x.size()) == 3:
            x = x.reshape(-1, self.seq_len, self.n_joint, 3).permute(0, 3, 1, 2)
        # print(x.size())
        identity = x
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        x = x + identity
        # print(x.size())
        x = x.permute(0, 2, 3, 1)
        # print(x.size())
        x = x.reshape(*size)
        # print(x.size())
        return {'refined_3d': x}
