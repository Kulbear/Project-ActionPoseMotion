"""
V1 is a sequential model where V2 is a residual learning version of V1.
"""
import torch.nn as nn


class RefineNetV1(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5):
        super(RefineNetV1, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
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
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5):
        super(RefineNetV2, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
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
