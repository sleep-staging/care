#%%
import torch.nn as nn
import torch.nn.functional as F
import torch

from .resnet1d import BaseNet
from config import Config
from typing import Type, Tuple


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
        self.scale = self.att_dim**-0.5

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

    def __init__(self, config: Type[Config]):
        super(encoder, self).__init__()
        self.time_model = BaseNet()
        self.attention = attention()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        time = self.time_model(x)

        time_feats = self.attention(time)

        return time_feats


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
        # x = x.reshape(x.shape[0],-1)
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

        self.q_encoder = encoder(config)
        self.k_encoder = encoder(config)

        for param_q, param_k in zip(self.q_encoder.parameters(),
                                    self.k_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

        self.q_proj = projection_head(config)
        self.k_proj = projection_head(config)

        for param_q, param_k in zip(self.q_proj.parameters(),
                                    self.k_proj.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

        self.config = config

    def forward(
            self, weak_data: torch.Tensor,
            strong_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        anchor = self.q_encoder(weak_data[:, (self.config.epoch_len //
                                              2):(self.config.epoch_len // 2) +
                                          1, :])
        anchor = self.q_proj(anchor)
        positive = self.k_encoder(
            strong_data[:, (self.config.epoch_len //
                            2):(self.config.epoch_len // 2) + 1, :])
        positive = self.k_proj(positive)

        return anchor, positive


class contrast_loss(nn.Module):
    """
    Class for the contrast loss

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters

    Methods:
    --------
        forward: torch.Tensor,torch.Tensor-> torch.Tensor,float

    """

    def __init__(self, config: Type[Config]):
        super(contrast_loss, self).__init__()

        self.config = config
        self.model = sleep_model(config)
        self.T = config.temperature

    def loss(self, anchor, positive, queue):

        # L2 normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        queue = F.normalize(queue, p=2, dim=1)

        # positive logits: Nx1, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [anchor, positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [anchor, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0],
                             dtype=torch.long).to(self.config.device)

        # loss
        loss = F.cross_entropy(logits, labels)

        return loss  # mean

    def forward(self, weak, strong, queue):
        anchor, positive = self.model(weak, strong)
        l1 = self.loss(anchor, positive, queue)

        return l1, positive


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

        self.eeg_encoder = encoder(config)
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
