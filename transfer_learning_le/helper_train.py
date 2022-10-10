import time, math
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy, cohen_kappa
from torchmetrics.functional import f1_score as f1
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import KFold
from utils.dataloader import TuneDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler


def run(config,name,test_subjects):

    if name=="simclr":
        from models.simclr.model import  ft_loss
        print("Hello")
    elif name=="mocov2":
        from models.mocov2.model import ft_loss
    elif name=="simsiam":
        from models.simsiam.model import ft_loss
    elif name=="simsiam_noBN":
        from models.simsiam_noBN.model import ft_loss
    elif name=="me":
        from models.me.model import ft_loss
    else:
        from models.simclr.model import ft_loss

    class sleep_pretrain(nn.Module):
    
        def __init__(self, config, name, test_subjects):
            super(sleep_pretrain, self).__init__()
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.config = config
            self.config.name = name
            self.ft_epochs = config.num_ft_epoch
    
            self.test_subjects = test_subjects
    
        def ft_fun(self, test_subjects_train, test_subjects_test):
    
            train_dl = DataLoader(
                TuneDataset(test_subjects_train),
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            test_dl = DataLoader(
                TuneDataset(test_subjects_test),
                batch_size=self.config.batch_size,
                shuffle=False,
            )
    
            sleep_eval = sleep_ft(
                self.config.exp_path + "/" + self.config.name + ".pt",
                self.config,
                train_dl,
                test_dl,
            )
            f1, kappa, bal_acc, acc = sleep_eval.fit()
    
            return f1, kappa, bal_acc, acc
    
        def do_kfold(self):
    
            kfold = KFold(n_splits=self.config.splits,
                          shuffle=True,
                          random_state=1234)
    
            k_acc, k_f1, k_kappa, k_bal_acc = 0, 0, 0, 0
            start = time.time()
            
            i = 0
            for train_idx, test_idx in kfold.split(self.test_subjects):
    
                test_subjects_train = [self.test_subjects[i] for i in train_idx]
                test_subjects_test = [self.test_subjects[i] for i in test_idx]
                test_subjects_train = [
                    rec for sub in test_subjects_train for rec in sub
                ]
                test_subjects_test = [
                    rec for sub in test_subjects_test for rec in sub
                ]
                
                i+=1
                print(f'Fold: {i}')
                self.config.split = i
                f1, kappa, bal_acc, acc = self.ft_fun(test_subjects_train, test_subjects_test)
                k_f1 += f1
                k_kappa += kappa
                k_bal_acc += bal_acc
                k_acc += acc
          
            pit = time.time() - start
            print(f"Took {int(pit // 60)} min:{int(pit % 60)} secs")
    
            return (
                k_f1 / self.config.splits,
                k_kappa / self.config.splits,
                k_bal_acc / self.config.splits,
                k_acc / self.config.splits,
            )
    
        def fit(self):
    
            f1, kappa, bal_acc, acc = self.do_kfold()
            print(
                f"F1: {f1},Kappa: {kappa},Bal Acc: {bal_acc},Acc: {acc}"
            )
    
    class sleep_ft(nn.Module):
    
        def __init__(self, chkpoint_pth, config, train_dl, valid_dl):
            super(sleep_ft, self).__init__()
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model = ft_loss(chkpoint_pth, config, self.device).to(self.device)
            self.config = config
            self.beta1 = config.beta1
            self.beta2 = config.beta2
            self.weight_decay = 3e-5
            self.batch_size = config.eval_batch_size
            self.loggr = wandb.init(project="carev2_linear_evaluation",notes='',save_code=True,entity='sleep-staging',name='split: '+str(self.config.split),group=self.config.name,job_type='split')
            self.config.exp_path + "/" + self.config.name + ".pt",
            self.criterion = nn.CrossEntropyLoss()
            self.train_ft_dl = train_dl
            self.valid_ft_dl = valid_dl
    
            self.max_f1 = torch.tensor(0).to(self.device)
            self.max_acc = torch.tensor(0).to(self.device)
            self.max_bal_acc = torch.tensor(0)
            self.max_kappa = torch.tensor(0).to(self.device)
    
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.config.lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.weight_decay,
            )
            self.ft_epoch = config.num_ft_epoch
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               mode="min",
                                               patience=10,
                                               factor=0.2)
    
        def train_dataloader(self):
            return self.train_dl
    
        def val_dataloader(self):
            return self.valid_dl
    
        def training_step(self, batch, batch_idx):
            data, y = batch
            data, y = data.float().to(self.device), y.long().to(self.device)
            outs = self.model(data)
            loss = self.criterion(outs, y)
            return loss
    
        def validation_step(self, batch, batch_idx):
            data, y = batch
            data, y = data.float().to(self.device), y.to(self.device)
            outs = self.model(data)
            loss = self.criterion(outs, y)
            acc = accuracy(outs, y)
            return {
                "loss": loss.detach(),
                "acc": acc,
                "preds": outs.detach(),
                "target": y.detach()
            }
    
        def validation_epoch_end(self, outputs,epoch):
    
            epoch_preds = torch.vstack([x for x in outputs["preds"]])
            epoch_targets = torch.hstack([x for x in outputs["target"]])
            epoch_loss = torch.hstack([torch.tensor(x)
                                       for x in outputs['loss']]).mean()
            epoch_acc = torch.hstack([torch.tensor(x)
                                      for x in outputs["acc"]]).mean()
            class_preds = epoch_preds.cpu().detach().argmax(dim=1)
            f1_sc = f1(epoch_preds, epoch_targets, average="macro", num_classes=5)
            kappa = cohen_kappa(epoch_preds, epoch_targets, num_classes=5)
            bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(),
                                              class_preds.cpu().numpy())
    
            self.loggr.log({
                'F1': f1_sc,
                'Kappa': kappa,
                'Bal Acc': bal_acc,
                'Acc': epoch_acc,
                'Epoch': epoch
            })
    
            if f1_sc > self.max_f1:
    
                #self.loggr.log({'Pretrain Epoch' : self.loggr.plot.confusion_matrix(probs=None,title=f'Pretrain Epoch :{self.pret_epoch+1}',
                #            y_true= epoch_targets.cpu().numpy(), preds= class_preds.numpy(),
                #            class_names= ['Wake', 'N1', 'N2', 'N3', 'REM'])})
    
                self.max_f1 = f1_sc
                self.max_kappa = kappa
                self.max_bal_acc = bal_acc
                self.max_acc = epoch_acc
    
            self.scheduler.step(epoch_loss)
    
            return epoch_loss
    
        def on_train_end(self):
            return self.max_f1, self.max_kappa, self.max_bal_acc, self.max_acc
    
        def fit(self):
    
            for ep in tqdm(range(self.ft_epoch), desc="Linear Evaluation"):
    
                # Training Loop
                self.model.train()
                ft_outputs = {"loss": [], "acc": [], "preds": [], "target": []}
                outputs = {'loss': []}
    
                for ft_batch_idx, ft_batch in enumerate(self.train_ft_dl):
                    loss = self.training_step(ft_batch, ft_batch_idx)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
    
                    outputs['loss'].append(loss.item())
                    loss = torch.hstack([torch.tensor(x)
                                         for x in outputs['loss']]).mean()
    
                # Validation Loop
                self.model.eval()
                with torch.no_grad():
                    for ft_batch_idx, ft_batch in enumerate(self.valid_ft_dl):
                        dct = self.validation_step(ft_batch, ft_batch_idx)
                        loss, acc, preds, target = (
                            dct["loss"],
                            dct["acc"],
                            dct["preds"],
                            dct["target"],
                        )
                        ft_outputs["loss"].append(loss.item())
                        ft_outputs["acc"].append(acc.item())
                        ft_outputs["preds"].append(preds)
                        ft_outputs["target"].append(target)
    
                    val_loss = self.validation_epoch_end(ft_outputs,ep)
    
    
            self.loggr.finish()
    
            return self.on_train_end()
    
    model = sleep_pretrain(config,name,test_subjects)
    model.fit()
