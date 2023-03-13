import torch
import torch.nn as nn

from models.ConvLSTMCell3d import ConvLSTMCell
from layers import SpatialTransformer
from models.unet_utils import *
class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan, size1, size2, size3):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        # BxCx1xDxWxH

        self.encoder1_conv = nn.Conv3d(in_channels=in_chan,
                                     out_channels=nf,
                                     kernel_size=(3, 3, 3),
                                     padding=(1, 1, 1))

        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.ConvLSTM3d1 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3,3,3),
                                        bias=True)
        self.ConvLSTM3d2 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3, 3, 3),
                                        bias=True)
        self.ConvLSTM3d3 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3, 3, 3),
                                        bias=True)
        self.ConvLSTM3d4 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3, 3, 3),
                                        bias=True)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.out = ConvOut(nf)

        self.transformer = SpatialTransformer((size1, size2, size3))




    def autoencoder(self, x, seq_len, rpm_x, rpm_y, future_step, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7):
        latent = []
        out = []
        # encoder
        e1 = []
        e2 = []
        e3 = []

        for t in range(seq_len):
            #print(rpm_x.shape, rpm_y.shape)
            h_t1 = self.encoder1_conv(x[:,t,...])
            down1 = self.down1(h_t1)

            h_t4, c_t4 = self.ConvLSTM3d1(input_tensor=down1,
                                   cur_state=[h_t4,c_t4])
            h_t5, c_t5 = self.ConvLSTM3d2(input_tensor = h_t4,
                                   cur_state = [h_t5,c_t5])
            h_t5 = torch.mul(h_t5,torch.squeeze(rpm_x[0,t-1]))
            # simple multiplication between rpm and feature
            encoder_vector = h_t5


        for t in range(future_step):

            h_t6, c_t6 = self.ConvLSTM3d3(input_tensor=encoder_vector,
                                   cur_state=[h_t6, c_t6])
            h_t7, c_t7 = self.ConvLSTM3d4(input_tensor=h_t6,
                                   cur_state=[h_t7, c_t7])
            h_t7 = torch.mul(h_t7, torch.squeeze(rpm_y[0,t]))
            # Simple multiplication between rpm and later phase features
            encoder_vector = h_t7
            latent += [h_t7]

        latent = torch.stack(latent,1)
        latent = latent.permute(0,2,1,3,4,5)
        timestep = latent.shape[2]

        output_img = []
        output_dvf = []
        # spatial transformer = transformer
        for i in range(timestep):
            output_ts = self.up1(latent[:,:,i,...])
            dvf = self.out(output_ts)
            warped_img = self.transformer(x[:,0,...],dvf)
            output_img += [warped_img]
            output_dvf += [dvf]

        output_img = torch.stack(output_img,1)
        output_dvf = torch.stack(output_dvf,1)
        output_img = output_img.permute(0,2,1,3,4,5)
        output_dvf = output_dvf.permute(0,2,1,3,4,5)

        return output_img, output_dvf


    def forward(self, x, rpm_x, rpm_y, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _,d, h, w = x.size()

        # initialize hidden states
        h_t4, c_t4 = self.ConvLSTM3d1.init_hidden(batch_size=b, image_size=(int(d // 2),int(h // 2),int(w // 2)))
        h_t5, c_t5 = self.ConvLSTM3d2.init_hidden(batch_size=b, image_size=(int(d // 2), int(h // 2), int(w // 2)))
        h_t6, c_t6 = self.ConvLSTM3d3.init_hidden(batch_size=b, image_size=(int(d // 2), int(h // 2), int(w // 2)))
        h_t7, c_t7 = self.ConvLSTM3d4.init_hidden(batch_size=b, image_size=(int(d // 2), int(h // 2), int(w // 2)))

        # autoencoder forward
        #outputs = self.autoencoder(x, seq_len, future_seq, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, m_t3, h_t4, c_t4, m_t4,
        #                           h_t5, c_t5, m_t5, h_t6, c_t6, m_t6, h_t7, c_t7, h_t8, c_t8)
        outputs = self.autoencoder(x, seq_len, rpm_x, rpm_y, future_seq, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7)

        return outputs
