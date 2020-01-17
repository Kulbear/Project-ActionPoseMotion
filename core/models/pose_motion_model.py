import torch
import torch.nn as nn


class PoseLifter(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=False, dropout_ratio=0.5):
        super(PoseLifter, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Linear(self.ipt_dim, self.hid_dim)
        self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)

    def forward(self, x):
        x = self.pre_rnn(x)
        output, hidden = self.rnn(x)
        out = self.post_rnn(output)
        return {'pose_3d': out, 'encoder_hidden': hidden}


class MotionGenerator(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128, n_layers=1, bidirectional=True, dropout_ratio=0.5):
        super(MotionGenerator, self).__init__()
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
        self.rnn = nn.LSTM(self.hid_dim, self.hid_dim * 2, self.n_layers,
                           bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)
        self.post_rnn = nn.Sequential(
            nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)
        )

    def forward(self, x, hidden):
        identity = x
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x, hidden)
        x = self.post_rnn(x)
        x = x + identity
        return {'motion_3d': x, 'decoder_hidden': hidden}


class Pose2MotNet(nn.Module):
    def __init__(self, encoder, decoder, device='cuda'):
        super(Pose2MotNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x, gt, teacher_forcing_ratio=0.5):
        batch_size, seq_len = gt.size()[:2]
        outputs = torch.zeros(batch_size, seq_len, self.decoder.opt_dim).to(self.device)

        # encoder_output = {'pose_3d': out, 'encoder_hidden': hidden}
        encoder_output = self.encoder(x)
        past_pose_sequence = encoder_output['pose_3d']
        hidden = encoder_output['encoder_hidden']
        # first input to the decoder is the last pose we observed
        output = past_pose_sequence[:, -1, :].unsqueeze(1)

        for t in range(seq_len):
            if t == 0:
                # ugly handle the lack of cell state in GRU
                prediction = self.decoder(output, (hidden, hidden))
            else:
                prediction = self.decoder(output, hidden)
            outputs[:, t, :] = prediction['motion_3d'].squeeze()
            hidden = prediction['decoder_hidden']
            # decide next input by teacher forcing ratio
            if torch.rand(1) < teacher_forcing_ratio:
                output = gt[:, t, :].unsqueeze(1)
            else:
                output = outputs[:, t, :].unsqueeze(1)

        return {
            'past_pose': past_pose_sequence,
            'future_motion': outputs
        }
