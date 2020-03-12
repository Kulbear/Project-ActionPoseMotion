"""
V1 is a sequential model where V2 is a residual learning version of V1.
"""
import torch.nn as nn


class KeypointRefineNet(nn.Module):
    def __init__(self, ipt_dim, opt_dim,
                 hid_dim=128, n_layers=1, bidirectional=False):
        super(KeypointRefineNet, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        hid_dim_factor = 2 if self.bidirectional else 1

        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.hid_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.hid_dim // 4, self.hid_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

        self.rnn = nn.GRU(self.hid_dim // 4, self.hid_dim // 4, self.n_layers,
                          bidirectional=self.bidirectional, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim // 4 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        identity = x
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x)
        x = self.post_rnn(x)
        x = x + identity
        return {'refined_2d': x}
