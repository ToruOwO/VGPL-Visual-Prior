import torch.nn as nn


def get_pos_diff(pos):
    """
    Calculate the temporal position displacement.

    Input:
        pos: (B, T, 3, N)
    Returns:
        pos_diff: (B, T-1, 3, N)
    """
    return pos[:, 1:, ...] - pos[:, :-1, ...]


class StackedConvBlock(nn.Module):
    """Stacked convolutional layers."""

    def __init__(self,
                 in_channel,
                 h_channel,
                 out_channel,
                 in_kernel_size=3,
                 out_kernel_size=3,
                 n_in_layers=2,
                 batchnorm=False):
        super(StackedConvBlock, self).__init__()
        input_modules = []
        for i in range(n_in_layers):
            if i == 0:
                input_modules.append(
                    nn.Conv2d(in_channel, h_channel, in_kernel_size, 1))
            else:
                input_modules.append(
                    nn.Conv2d(h_channel, h_channel, in_kernel_size, 1))
            if batchnorm:
                input_modules.append(nn.BatchNorm2d(h_channel))
            input_modules.append(nn.ReLU())
        self.input_net = nn.Sequential(*input_modules)

        output_modules = []
        output_modules.append(
            nn.Conv2d(h_channel, out_channel, out_kernel_size, 2))
        if batchnorm:
            output_modules.append(nn.BatchNorm2d(out_channel))
        output_modules.append(nn.ReLU())
        self.output_net = nn.Sequential(*output_modules)

    def forward(self, x):
        z = self.input_net(x)
        y = self.output_net(z)
        return y


class Encoder(nn.Module):
    """Encoder."""

    def __init__(self, n_frames=1, batchnorm=True):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            StackedConvBlock(3*n_frames, 32, 64, batchnorm=batchnorm),  # (57, 77)
            StackedConvBlock(64, 64, 128, batchnorm=batchnorm),  # (26, 36)
            StackedConvBlock(128, 128, 256, batchnorm=batchnorm),  # (10, 15)
            StackedConvBlock(256, 256, 512, batchnorm=batchnorm),  # (2, 5)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Predictor."""

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_particles=192,
                 n_frames=1,
                 n_group_classes=3):
        super(Predictor, self).__init__()
        self._n_particles = n_particles
        self._n_frames = n_frames
        self._n_group_classes = n_group_classes

        self.pos_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, 3*n_frames*n_particles))

        self.group_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, n_group_classes*n_frames*n_particles))

    def forward(self, x):
        pred_pos = self.pos_head(x).view(
            -1, self._n_frames, 3, self._n_particles)
        pred_grp = self.group_head(x).view(
            -1, self._n_frames, self._n_particles, self._n_group_classes)
        return pred_pos, pred_grp


class RecurrentPredictor(nn.Module):
    """Recurrent predictor."""

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_particles=192,
                 n_group_classes=3):
        super(RecurrentPredictor, self).__init__()
        self._n_particles = n_particles
        self._n_group_classes = n_group_classes

        self.core = nn.LSTM(
            input_size, hidden_size, batch_first=True)  # rnn cell
        self.pos_head = nn.Linear(hidden_size, 3*n_particles)
        self.group_head = nn.Linear(hidden_size, n_group_classes*n_particles)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, _ = x.shape
        x, _ = self.core(x)  # B, T, h
        pred_pos = self.pos_head(x.reshape(B*T, -1)).reshape(
            B, T, 3, self._n_particles)
        pred_grp = self.group_head(x.reshape(B*T, -1)).reshape(
            B, T, self._n_particles, self._n_group_classes)
        return pred_pos, pred_grp


class ConvTempEncoder(nn.Module):
    """Convolutional temporal encoder block from Kanazawa et al."""

    def __init__(self, n_filters, kernel_width, n_blocks=1):
        super(ConvTempEncoder, self).__init__()
        self._n_blocks = n_blocks

        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters,
                          kernel_size=(kernel_width, 1),
                          padding=(1, 0)),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters,
                          kernel_size=(kernel_width, 1),
                          padding=(1, 0)),
            ))

    def forward(self, x):
        """
        x: (B, T, C)
        """
        x_ = x.transpose(1, 2).unsqueeze(-1)
        for b in self.blocks:
            x_ = b(x_) + x_
        return x_.squeeze().transpose(1, 2)


class TempEncodingPredictor(nn.Module):
    """
    Predictor with temporal encoder.
    Different from Predictor, input to this model has shape (B, T, D).
    """

    def __init__(self,
                 input_size,
                 embedding_size,
                 n_frames,
                 n_particles,
                 n_group_classes,
                 hidden_size,
                 conv_temp_encoder=False,
                 kernel_width=3,
                 n_blocks=1):
        super(TempEncodingPredictor, self).__init__()
        self._input_size = input_size
        self._embedding_size = embedding_size
        self._conv_temp_encoder = conv_temp_encoder
        self._n_frames = n_frames
        self._n_particles = n_particles
        self._n_group_classes = n_group_classes
        if conv_temp_encoder:
            self._hidden_size = n_frames*embedding_size
        else:
            # self._hidden_size = hidden_size
            self._hidden_size = n_frames*embedding_size

        self.embedding_layer = nn.Linear(input_size, embedding_size)
        if conv_temp_encoder:
            if kernel_width > n_frames:
                raise ValueError(
                    'Kernel width cannot be smaller than number of frames.')
            self.temp_encoder = ConvTempEncoder(
                embedding_size, kernel_width, n_blocks)
        else:
            self.temp_encoder = nn.Linear(n_frames*embedding_size, hidden_size)

        # temporal encoding predictors output the frame for the last step
        self.pos_head = nn.Linear(
            self._hidden_size, 3*n_particles)
        self.grp_head = nn.Linear(
            self._hidden_size, n_group_classes*n_particles)

    def forward(self, x):
        B, T, D = x.shape
        x = self.embedding_layer(
            x.view(B*T, D)).view(B, T, self._embedding_size)
        if self._conv_temp_encoder:
            x = self.temp_encoder(x)
        # else:
        #     x = self.temp_encoder(x.view(B, T*self._embedding_size))

        pred_pos = self.pos_head(
            x.contiguous().view(B, self._hidden_size)).view(
            B, 1, 3, self._n_particles)
        pred_grp = self.grp_head(
            x.contiguous().view(B, self._hidden_size)).view(
            B, 1, self._n_particles, self._n_group_classes)

        return pred_pos, pred_grp


class PointSetNet(nn.Module):
    """Full Model."""

    def __init__(self,
                 n_frames,
                 pred_hidden,
                 n_particles,
                 n_group_classes,
                 batchnorm=True,
                 single_out=False,
                 recur_pred=False,
                 use_temp_encoder=False,
                 conv_temp_encoder=False,
                 temp_embedding_size=1024,
                 conv_temp_kernel_width=3,
                 conv_temp_n_blocks=1):
        super(PointSetNet, self).__init__()
        self._recur_pred = recur_pred
        self._single_out = single_out
        self._use_temp_encoder = use_temp_encoder
        self._conv_temp_encoder = conv_temp_encoder

        if recur_pred:
            self.encoder = Encoder(1, batchnorm)
            self.predictor = RecurrentPredictor(
                512 * 2 * 5,
                pred_hidden,
                n_particles=n_particles,
                n_group_classes=n_group_classes)
        else:
            if use_temp_encoder:
                self.encoder = Encoder(1, batchnorm)
                self.predictor = TempEncodingPredictor(
                    512 * 2 * 5,
                    temp_embedding_size,
                    n_particles=n_particles,
                    n_frames=n_frames,
                    n_group_classes=n_group_classes,
                    hidden_size=pred_hidden,
                    conv_temp_encoder=conv_temp_encoder,
                    kernel_width=conv_temp_kernel_width,
                    n_blocks=conv_temp_n_blocks)
            else:
                self.encoder = Encoder(n_frames, batchnorm)

                if single_out:
                    n_output_frames = 1
                else:
                    n_output_frames = n_frames
                self.predictor = Predictor(
                    512 * 2 * 5,
                    pred_hidden,
                    n_particles=n_particles,
                    n_frames=n_output_frames,
                    n_group_classes=n_group_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        if not self._recur_pred:
            if self._use_temp_encoder:
                x = x.view(B*T, C, H, W)
                z = self.encoder(x).view(B, T, 512*2*5)
                return self.predictor(z)
            else:
                x = x.view(B, T*C, H, W)
                z = self.encoder(x)
                return self.predictor(z.view(z.shape[0], -1))
        else:
            x = x.view(B*T, C, H, W)
            z = self.encoder(x).view(B, T, 512*2*5)
            pp, pg = self.predictor(z)

            if self._single_out:
                # return the last time step
                return pp[:, -2:-1, ...], pg[:, -2:-1, ...]
            return pp, pg
