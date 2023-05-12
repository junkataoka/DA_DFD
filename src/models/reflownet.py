import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity


class EncoderDecoderConvLSTM(nn.Module):

    def __init__(self, nf, in_chan, seq_len):
        super(EncoderDecoderConvLSTM, self).__init__()

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf*seq_len,  # nf + 1
                                               hidden_dim=nf*seq_len,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf*seq_len,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.linear1_src = nn.Linear(17500, 2400)
        self.linear2_src = nn.Linear(2400, 600)
        self.linear3_src = nn.Linear(600, 7)
        self.linear1_tar = nn.Linear(17500, 2400)
        self.linear2_tar = nn.Linear(2400, 600)
        self.linear3_tar = nn.Linear(600, 7)
        self.relu = nn.ReLU(inplace=True)
        self.nf = nf

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t3, c_t3):

        output_enc_h = []
        output_dec_h = []

        # encoder
        b = x.shape[0]
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :, :],
                                               cur_state=[h_t, c_t])

            output_enc_h += [h_t]

        encoder_vector = torch.stack(output_enc_h)
        encoder_vector = encoder_vector.reshape(b,  self.nf * seq_len, 50, 50)

        # decoder
        for t in range(future_step):

            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])
            output_dec_h += [h_t3]

        output_dec_h = self.stack_permute(output_dec_h)
        output_last = self.decoder_CNN(output_dec_h)
        output_last = output_last.view((b, -1))

        return output_last

    def bottleneck_src(self, output_last):

        ln = self.linear1_src(output_last)
        ln_relu = self.relu(ln)
        ln_last = self.linear2_src(ln_relu)
        ln_last_relu = self.relu(ln_last)
        pred = self.linear3_src(ln_last_relu)

        return pred, ln_last, ln

    def bottleneck_tar(self, output_last):

        ln = self.linear1_tar(output_last)
        ln_relu = self.relu(ln)
        ln_last = self.linear2_tar(ln_relu)
        ln_last_relu = self.relu(ln_last)
        pred = self.linear3_tar(ln_last_relu)

        return pred, ln_last, ln

    def stack_permute(self, vec):
        vec = torch.stack(vec, 1)
        vec = vec.permute(0, 2, 1, 3, 4)
        return vec

    def forward(self, x, future_step, domain):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)
            batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
            )
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
            )

        output_last = self.autoencoder(
            x, seq_len, future_step, h_t, c_t, h_t3, c_t3
            )

        if domain == "src":
            outputs, feat1, feat2 = self.bottleneck_src(output_last)

        elif domain == "tar":
            outputs, feat1, feat2 = self.bottleneck_tar(output_last)

        return outputs, feat1, feat2


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.stride = 1

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.layernorm = nn.LayerNorm([self.input_dim, 50, 50])

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        input_tensor = self.layernorm(input_tensor)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
            )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size, self.hidden_dim, height, width, 
                device=self.conv.weight.device),
            torch.zeros(
                batch_size, self.hidden_dim, height, width,
                device=self.conv.weight.device)
                )

def da_cos_loss(src_x, tar_x):
    sim_mat = pairwise_cosine_similarity(src_x, tar_x)
    pos_sim_mat = 0.5 * (1 + sim_mat)
    return pos_sim_mat.mean()

