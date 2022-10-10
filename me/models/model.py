import torch.nn as nn
import torch.nn.functional as F
import torch

from .resnet1d import BaseNet
from config import Config
from typing import Type, Tuple
from .tfr import Transformer


class attention(nn.Module):
    """
    Class for the attention module of the model

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the attention module

    """
    
    def __init__(self):
        super(attention, self).__init__()
        self.att_dim = 256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e * self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


class encoder(nn.Module):
    """
    Class for the encoder of the model that contains both time encoder and spectrogram encoder

    Attributes:
    -----------
        config: Config object
            Configuration object specifying the model hyperparameters

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the encoder

    """

    def __init__(self):
        super(encoder, self).__init__()
        self.model = BaseNet()
        self.attention = attention()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.model(x)
        x = self.attention(x)
        return x


class projection_head(nn.Module):
    """
    Class for the projection head of the model

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters

        input_dim: int, optional
            input dimension of the model

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the projection head

    """

    def __init__(self, config: Type[Config], input_dim: int = 256):
        super(projection_head, self).__init__()
        self.config = config
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, config.tc_hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(config.tc_hidden_dim, config.tc_hidden_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection_head(x)
        return x


class sleep_model(nn.Module):

    """
    Class for the sleep model

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters

    Methods:
    --------
        forward: torch.Tensor,torch.Tensor -> torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor
            returns the time, spectrogram, and fusion features for both weak and strong inputs

    """

    def __init__(self, config: Type[Config]):
        super(sleep_model, self).__init__()

        self.eeg_encoder = encoder()
        self.curr_weak_pj = projection_head(config)
        self.curr_strong_pj = projection_head(config)
        self.surr_weak_pj = projection_head(config)
        self.surr_strong_pj = projection_head(config)

        self.config = config
        self.tfmr = Transformer(256, 4, 4, 256, dropout=0.1)

    def forward(self, weak_dat: torch.Tensor, strong_dat: torch.Tensor):

        weak_eeg_dat = weak_dat.float()
        strong_eeg_dat = strong_dat.float()

        weak_surr_feats = []
        strong_surr_feats = []

        for i in range(self.config.epoch_len):
            weak_surr_feats.append(self.eeg_encoder(weak_eeg_dat[:, i : i + 1, :]))
            strong_surr_feats.append(self.eeg_encoder(strong_eeg_dat[:, i : i + 1, :]))

        ep = torch.randint(self.config.epoch_len, (1,)).item()
        weak_curr_feats = weak_surr_feats[ep]
        strong_curr_feats = strong_surr_feats[ep]

        weak_surr_feats = torch.stack(weak_surr_feats, dim=1)
        strong_surr_feats = torch.stack(strong_surr_feats, dim=1)

        weak_surr_feats = self.tfmr(weak_surr_feats)
        strong_surr_feats = self.tfmr(strong_surr_feats)

        weak_curr_feats, strong_curr_feats = self.curr_weak_pj(
            weak_curr_feats
        ), self.curr_strong_pj(strong_curr_feats)

        weak_surr_feats, strong_surr_feats = self.surr_weak_pj(
            weak_surr_feats
        ), self.surr_strong_pj(strong_surr_feats)

        return weak_curr_feats, weak_surr_feats, strong_curr_feats, strong_surr_feats


class contrast_loss(nn.Module):
    """
    Class for the contrast loss

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters

    Methods:
    --------
        forward: torch.Tensor,torch.Tensor-> torch.Tensor,float,float,float,float

    """

    def __init__(self, config: Type[Config]):

        super(contrast_loss, self).__init__()
        self.model = sleep_model(config)
        self.T = config.temperature
        self.config = config

    def loss(self, out_1: torch.Tensor, out_2: torch.Tensor):
        # L2 normalize
        out_1 = F.normalize(out_1, p=2, dim=1)
        out_2 = F.normalize(out_2, p=2, dim=1)

        out = torch.cat([out_1, out_2], dim=0)  # 2B, 128
        N = out.shape[0]

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())  # 2B, 2B
        sim = torch.exp(cov / self.T)  # 2B, 2B

        # Negative similarity matrix
        mask = ~torch.eye(N, device=sim.device).bool()
        neg = sim.masked_select(mask).view(N, -1).sum(dim=-1)

        # Positive similarity matrix
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.T)
        pos = torch.cat([pos, pos], dim=0)  # 2B
        loss = -torch.log(pos / neg).mean()
        return loss

    def forward(
        self, weak: torch.Tensor, strong: torch.Tensor
    ):

        (
            top_curr,
            top_surr,
            bot_curr,
            bot_surr,
        ) = self.model(weak, strong)

        l1 = self.loss(top_curr, bot_curr)
        l2 = self.loss(top_surr, bot_surr)

        l3 = self.loss(top_curr, bot_surr)
        l4 = self.loss(top_surr, bot_curr)

        tot_loss = (l1 + l2) + self.config.lambda1 * (l3 + l4)

        return tot_loss


class ft_loss(nn.Module):
    """
    Class for the linear evaluation module for time encoder features

    Attributes:
    ----------
        chkpoint_pth: str
            Path to the checkpoint file
        config: Config object
            Configuration object containing hyperparameters
        device: str
            Device to use for training

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            Computes the forward pass

    """

    def __init__(self, chkpoint_pth: str, config: Type[Config], device: str):

        super(ft_loss, self).__init__()

        self.eeg_encoder = encoder()
        chkpoint = torch.load(chkpoint_pth, map_location=device)
        eeg_dict = chkpoint["eeg_model_state_dict"]
        self.eeg_encoder.load_state_dict(eeg_dict)

        for p in self.eeg_encoder.parameters():
            p.requires_grad = False

        self.lin = nn.Linear(256, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.eeg_encoder(x)
        x = self.lin(x)

        return x
