import torch

class Config(object):

    def __init__(self,wandb=None):

        self.src_path = "/scratch/SLEEP_data/"
        self.exp_path = "."
        self.wandb = wandb
        self.batch_size = 128
        self.eval_batch_size = 512
        self.eval_early_stopping = 15

        self.degree = 0.05
        self.mask_max_points = 200
        self.mask_min_points = 50 

        self.proj_dim = 128

        # loss parameters
        self.temperature = 1

        # optimizer paramters
        self.optimizer = "adam"
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0003

        self.num_epoch = 250
        self.num_ft_epoch = 100

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drop_last = True
