"""
V1 is a sequential model where V2 is a residual learning version of V1.
"""
import torch.nn as nn


class TrajRefinementModule(nn.Module):
    def __init__(self, ipt_dim, opt_dim,
                 hid_dim=128, n_layers=1, bidirectional=False,
                 dropout_ratio=0.5, size=(64, 28, 45),
                 include_lie_repr=False):
        super(TrajRefinementModule, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        # layers
        if include_lie_repr:
            print('Large Refinement!!')
            self.pre_rnn = nn.Sequential(
                nn.Linear(self.ipt_dim, self.hid_dim),
                nn.ReLU(),
                nn.Linear(self.hid_dim, self.hid_dim * 2)
            )

            self.rnn = nn.GRU(self.hid_dim * 2, self.hid_dim * 2, self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)
        else:
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


class ResTrajRefinementModule(nn.Module):
    def __init__(self, ipt_dim, opt_dim,
                 hid_dim=128, n_layers=1, bidirectional=False,
                 dropout_ratio=0.5, size=(64, 28, 45),
                 include_lie_repr=False):
        super(ResTrajRefinementModule, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        if include_lie_repr:
            print('Large Refinement!!')
            self.pre_rnn = nn.Sequential(
                nn.Linear(self.ipt_dim, self.hid_dim),
                nn.ReLU(),
                nn.Linear(self.hid_dim, self.hid_dim * 2)
            )

            self.rnn = nn.GRU(self.hid_dim * 2, self.hid_dim * 2, self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)
        else:
            self.pre_rnn = nn.Sequential(
                nn.Linear(self.ipt_dim, self.hid_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hid_dim // 2, self.hid_dim)
            )

            self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        # print(x.size())
        # print(self.pre_rnn)
        identity = x
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x)
        x = self.post_rnn(x)
        x = x + identity
        return {'refined_3d': x}


# TODO: the current implementation can only handle past + future = 28
class ResConvTrajRefinementModule(nn.Module):
    def __init__(self, *args, size=(64, 28, 45), **kwargs):
        super(ResConvTrajRefinementModule, self).__init__()

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
        size = x.size()  # for unknown batch size
        if len(x.size()) == 3:
            x = x.reshape(-1, self.seq_len, self.n_joint, 3).permute(0, 3, 1, 2)
        identity = x
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        x = x + identity
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(*size)
        return {'refined_3d': x}


# TODO: the current implementation can only handle past + future = 28
class EnsTrajRefinementModule(nn.Module):
    def __init__(self, ipt_dim, opt_dim,
                 hid_dim=128, n_layers=1, bidirectional=False,
                 dropout_ratio=0.5, size=(64, 28, 45)):
        super(EnsTrajRefinementModule, self).__init__()

        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.size = size
        self.seq_len = size[1]
        self.n_joint = size[2] // 3
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
        identity = x
        x1 = self.pre_rnn(x)
        x1, hidden = self.rnn(x1)
        x1 = self.post_rnn(x1)
        x1 = x1 + identity

        size = x.size()
        if len(size) == 3:
            x2 = x.reshape(-1, self.seq_len, self.n_joint, 3).permute(0, 3, 1, 2)
        identity = x2
        x2 = self.encoder(x2)
        x2 = self.latent(x2)
        x2 = self.decoder(x2)
        x2 = x2 + identity
        x2 = x2.permute(0, 2, 3, 1)
        x2 = x2.reshape(*size)
        return {'refined_3d': (x1 + x2) / 2}
