import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu

LOG_COV_MAX = 2
LOG_COV_MIN = -1

class VeryRecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        conv_channels = 32
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, conv_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU()
        )
        ae_dim = 512
        lstm_dim = 2048
        self.lstm_dim = lstm_dim
        img_h = 5
        flat_inter_img_dim = img_h * img_h * conv_channels
        self.flat_inter_img_dim = flat_inter_img_dim
        act_dim = 64

        self.conv_channels = conv_channels
        self.img_h = img_h
        
        self.lstm_act_proc_fc = nn.Linear(4, act_dim, bias=True)
        self.recon_act_proc_fc = nn.Linear(4, act_dim, bias=True)

        self.attention_lstm = nn.LSTMCell(
            lstm_dim + act_dim, flat_inter_img_dim
        )

        self.fc_encoder = nn.Sequential(
            nn.Linear(flat_inter_img_dim + act_dim, ae_dim, bias=False),
            nn.BatchNorm1d(ae_dim),
            nn.ReLU(),
            # nn.Linear(ae_dim, ae_dim, bias=False),
            # nn.BatchNorm1d(ae_dim),
            # nn.ReLU(),
            # nn.Linear(ae_dim, ae_dim, bias=False),
            # nn.BatchNorm1d(ae_dim),
            # nn.ReLU(),
            # nn.Linear(ae_dim, ae_dim, bias=False),
            # nn.BatchNorm1d(ae_dim),
            # nn.ReLU()
        )
        self.lstm = nn.LSTMCell(
            ae_dim, lstm_dim, bias=True
        )
        self.conv_decoder = nn.Sequential(
            # nn.ConvTranspose2d(conv_channels, conv_channels, 4, stride=2, padding=1, output_padding=0, bias=False),
            # nn.BatchNorm2d(conv_channels),
            # nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            # nn.ConvTranspose2d(conv_channels, conv_channels, 4, stride=2, padding=1, output_padding=0, bias=False),
            # nn.BatchNorm2d(conv_channels),
            # nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )
        self.mean_decoder = nn.Sequential(
            nn.Conv2d(conv_channels, 3, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.log_cov_decoder = nn.Sequential(
            nn.Conv2d(conv_channels, 3, 1, stride=1, padding=0, bias=True),
        )

    
    def forward(self, obs_batch, act_batch, prev_h_batch, prev_c_batch):
        lstm_act_proc = self.lstm_act_proc_fc(act_batch)
        recon_act_proc = self.recon_act_proc_fc(act_batch)
        
        batch_size = obs_batch.size(0)
        att_prev_h_batch = Variable(torch.zeros(batch_size, self.flat_inter_img_dim))
        att_prev_c_batch = Variable(torch.zeros(batch_size, self.flat_inter_img_dim))
        if ptu.gpu_enabled():
            att_prev_h_batch = att_prev_h_batch.cuda()
            att_prev_c_batch = att_prev_c_batch.cuda()
        
        self.reg_loss = 0.
        
        att_input = torch.cat([prev_h_batch, recon_act_proc], 1)
        for _ in range(4):
            att_prev_h_batch, att_prev_c_batch = self.attention_lstm(att_input, (att_prev_h_batch, att_prev_c_batch))

        hidden = att_prev_h_batch.view(obs_batch.size(0), self.conv_channels, self.img_h, self.img_h)
        hidden = self.conv_decoder(hidden)
        recon = self.mean_decoder(hidden)
        log_cov = self.log_cov_decoder(hidden)
        log_cov = torch.clamp(log_cov, LOG_COV_MIN, LOG_COV_MAX)

        enc = self.conv_encoder(obs_batch)
        enc = enc.view(obs_batch.size(0), -1)
        enc = self.fc_encoder(torch.cat([enc, lstm_act_proc], 1))
        prev_h_batch, prev_c_batch = self.lstm(enc, (prev_h_batch, prev_c_batch))

        return recon, log_cov, prev_h_batch, prev_c_batch
