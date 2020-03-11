import torch
import torch.nn as nn


class PoseLifter(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128,
                 n_layers=1, bidirectional=False, dropout_ratio=0.5,
                 use_lie_algebra=False):
        super(PoseLifter, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.use_lie_algebra = use_lie_algebra
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Linear(self.ipt_dim, self.hid_dim)
        self.rnn = nn.GRU(self.hid_dim, self.hid_dim * 2, self.n_layers,
                          bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        # euler repr shape = # joint * 6
        self.post_rnn = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)
        if self.use_lie_algebra:
            # lie repr shape = # joint * 6
            self.post_rnn_lie = nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim * 2)

    def forward(self, x):
        x = self.pre_rnn(x)
        output, hidden = self.rnn(x)
        out = self.post_rnn(output)
        if self.use_lie_algebra:
            out_lie = self.post_rnn_lie(output)
            return {'pose_3d': out, 'pose_lie': out_lie, 'encoder_hidden': hidden}
        else:
            return {'pose_3d': out, 'encoder_hidden': hidden}


class MotionGenerator(nn.Module):
    def __init__(self, ipt_dim, opt_dim, hid_dim=128,
                 n_layers=1, bidirectional=True, dropout_ratio=0.5,
                 use_lie_algebra=False):
        super(MotionGenerator, self).__init__()
        # config
        self.ipt_dim = ipt_dim
        self.hid_dim = hid_dim
        self.opt_dim = opt_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_ratio = dropout_ratio
        self.use_lie_algebra = use_lie_algebra
        hid_dim_factor = 2 if self.bidirectional else 1

        # layers
        self.pre_rnn = nn.Sequential(
            nn.Linear(self.ipt_dim, self.hid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hid_dim // 2, self.hid_dim)
        )
        if self.use_lie_algebra:
            self.rnn = nn.LSTM(self.hid_dim * 2, self.hid_dim * 2, self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.hid_dim, self.hid_dim * 2, self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout_ratio, batch_first=True)

        self.post_rnn = nn.Sequential(
            nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim)
        )
        if self.use_lie_algebra:
            self.pre_rnn_lie = nn.Sequential(
                nn.Linear(self.ipt_dim * 2, self.hid_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hid_dim // 2, self.hid_dim)
            )
            self.post_rnn_lie = nn.Sequential(
                nn.Linear(self.hid_dim * 2 * hid_dim_factor, self.opt_dim * 2)
            )

    def plain_forward(self, x, hidden):
        identity = x
        x = self.pre_rnn(x)
        x, hidden = self.rnn(x, hidden)
        x = self.post_rnn(x)
        x = x + identity
        return {'motion_3d': x, 'motion_lie': x, 'decoder_hidden': hidden}

    def lie_forward(self, x, hidden):
        x, x_lie = x
        identity_x = x
        identity_x_lie = x_lie
        x = self.pre_rnn(x)
        x_lie = self.pre_rnn_lie(x_lie)
        rnn_ipt = torch.cat((x_lie, x), dim=2)
        latent, hidden = self.rnn(rnn_ipt, hidden)
        x = self.post_rnn(latent)
        x = x + identity_x
        x_lie = self.post_rnn_lie(latent)
        x_lie = x_lie + identity_x_lie
        return {'motion_3d': x, 'motion_lie': x_lie, 'decoder_hidden': hidden}

    def forward(self, x, hidden):
        if self.use_lie_algebra:
            return self.lie_forward(x, hidden)
        else:
            return self.plain_forward(x, hidden)


class Pose2MotNet2D(nn.Module):
    def __init__(self, encoder, decoder, refiner_2d=None, device='cuda', use_lie_algebra=False):
        super(Pose2MotNet2D, self).__init__()
        self.refiner_2d = refiner_2d
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.use_lie_algebra = use_lie_algebra

    def forward(self, x, gt, gt_lie=None,
                teacher_forcing_ratio=0.5,
                pos_loss_on=True, mot_loss_on=True):
        batch_size, seq_len = gt.size()[:2]
        outputs = torch.zeros(batch_size, seq_len, self.decoder.opt_dim).to(self.device)
        outputs_lie = torch.zeros(batch_size, seq_len, self.decoder.opt_dim * 2).to(self.device)

        encoder_output = self.encoder(x)
        past_pose_sequence = encoder_output['pose_3d']
        hidden = encoder_output['encoder_hidden']
        past_pose_lie_sequence = encoder_output.get('pose_lie', None)
        if not mot_loss_on:
            return {
                'past_pose': past_pose_sequence,
                'past_pose_lie': past_pose_lie_sequence,
                'future_motion': None,
                'future_motion_lie': None
            }
        # first input to the decoder is the last pose we observed
        output = past_pose_sequence[:, -1, :].unsqueeze(1)

        if self.use_lie_algebra:
            output_lie = past_pose_lie_sequence[:, -1, :].unsqueeze(1)
            output = (output, output_lie)

        for t in range(seq_len):
            if t == 0:
                # ugly handle the lack of cell state in GRU
                prediction = self.decoder(output, (hidden, hidden))
            else:
                prediction = self.decoder(output, hidden)
            # print(prediction['motion_3d'].size())
            # print(prediction['motion_lie'].size())
            outputs[:, t, :] = prediction['motion_3d'].squeeze()
            if self.use_lie_algebra:
                outputs_lie[:, t, :] = prediction['motion_lie'].squeeze()
            hidden = prediction['decoder_hidden']

            # decide next input by teacher forcing ratio
            if torch.rand(1) < teacher_forcing_ratio:
                output = gt[:, t, :].unsqueeze(1)
                if self.use_lie_algebra:
                    output_lie = gt_lie[:, t, :].unsqueeze(1)
                    output = (output, output_lie)
            else:
                output = outputs[:, t, :].unsqueeze(1)
                if self.use_lie_algebra:
                    output_lie = outputs_lie[:, t, :].unsqueeze(1)
                    output = (output, output_lie)

        if not pos_loss_on:
            # acutally no need?
            return {
                'past_pose': None,
                'past_pose_lie': None,
                'future_motion': outputs,
                'future_motion_lie': outputs_lie
            }
        else:
            return {
                'past_pose': past_pose_sequence,
                'past_pose_lie': past_pose_lie_sequence,
                'future_motion': outputs,
                'future_motion_lie': outputs_lie
            }


class Pose2MotNet(nn.Module):
    def __init__(self, encoder, decoder, device='cuda', use_lie_algebra=False):
        super(Pose2MotNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.use_lie_algebra = use_lie_algebra

    def forward(self, x, gt, gt_lie=None,
                teacher_forcing_ratio=0.5,
                pos_loss_on=True, mot_loss_on=True):
        batch_size, seq_len = gt.size()[:2]
        outputs = torch.zeros(batch_size, seq_len, self.decoder.opt_dim).to(self.device)
        outputs_lie = torch.zeros(batch_size, seq_len, self.decoder.opt_dim * 2).to(self.device)
        # encoder_output = {'pose_3d': out, 'encoder_hidden': hidden}
        encoder_output = self.encoder(x)
        past_pose_sequence = encoder_output['pose_3d']
        hidden = encoder_output['encoder_hidden']
        past_pose_lie_sequence = encoder_output.get('pose_lie', None)
        if not mot_loss_on:
            return {
                'past_pose': past_pose_sequence,
                'past_pose_lie': past_pose_lie_sequence,
                'future_motion': None,
                'future_motion_lie': None
            }
        # first input to the decoder is the last pose we observed
        output = past_pose_sequence[:, -1, :].unsqueeze(1)

        if self.use_lie_algebra:
            output_lie = past_pose_lie_sequence[:, -1, :].unsqueeze(1)
            output = (output, output_lie)

        for t in range(seq_len):
            if t == 0:
                # ugly handle the lack of cell state in GRU
                prediction = self.decoder(output, (hidden, hidden))
            else:
                prediction = self.decoder(output, hidden)
            # print(prediction['motion_3d'].size())
            # print(prediction['motion_lie'].size())
            outputs[:, t, :] = prediction['motion_3d'].squeeze()
            if self.use_lie_algebra:
                outputs_lie[:, t, :] = prediction['motion_lie'].squeeze()
            hidden = prediction['decoder_hidden']

            # decide next input by teacher forcing ratio
            if torch.rand(1) < teacher_forcing_ratio:
                output = gt[:, t, :].unsqueeze(1)
                if self.use_lie_algebra:
                    output_lie = gt_lie[:, t, :].unsqueeze(1)
                    output = (output, output_lie)
            else:
                output = outputs[:, t, :].unsqueeze(1)
                if self.use_lie_algebra:
                    output_lie = outputs_lie[:, t, :].unsqueeze(1)
                    output = (output, output_lie)

        if not pos_loss_on:
            # acutally no need?
            return {
                'past_pose': None,
                'past_pose_lie': None,
                'future_motion': outputs,
                'future_motion_lie': outputs_lie
            }
        else:
            return {
                'past_pose': past_pose_sequence,
                'past_pose_lie': past_pose_lie_sequence,
                'future_motion': outputs,
                'future_motion_lie': outputs_lie
            }
