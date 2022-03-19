import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune

from sklearn.model_selection import KFold

import wandb

import uuid
import datetime
import os

from src.data.make_dataset import ImageDataset
from src.models.model import Net

TUNE_ORIG_WORKING_DIR = os.getcwd()


class MLPipe():


    def __init__(self, config_dict=None):
        
        self.__parse_config_dict__(config_dict)

        self.__set_hp_params__()


    def __parse_config_dict__(self, config_dict):

        self.DATA = config_dict['data']
        self.TRAINING_HP = config_dict['training']
        self.OPTIMIZER = config_dict['optimizer']
        self.TUNING = config_dict['tuning']
        self.WANDB = config_dict['wandb']
        self.PROJECT = config_dict['project']
        self.VALIDATION = config_dict['validation']

        try:
            self.PROJECT['experiment']
        except KeyError:
            self.PROJECT['experiment'] = str(datetime.datetime.now())+'-'+str(uuid.uuid4().hex)

        self.channels = (1 if self.DATA['greyscale'] else 3)


    def __set_hp_params__(self):
        self.is_hp = False
        self.hp = {'max_epochs': self.TRAINING_HP['max_epochs']}

        if isinstance(self.TRAINING_HP['batch_size'], list):
            self.hp['batch_size'] = tune.choice(self.TRAINING_HP['batch_size'])
            self.is_hp = True
        else:
            self.hp['batch_size'] = self.TRAINING_HP['batch_size']

        if isinstance(self.OPTIMIZER['learning_rate'], list):
            self.hp['lr'] = tune.loguniform(self.OPTIMIZER['learning_rate'][0], self.OPTIMIZER['learning_rate'][1])
            self.is_hp = True
        else:
            self.hp['lr'] = self.OPTIMIZER['learning_rate']

        """
        if isinstance(self.OPTIMIZER['step_size'], list):
            self.hp['step_size'] = tune.choice(self.OPTIMIZER['step_size'])
            self.is_hp = True
        else:
           self.hp['step_size'] = self.OPTIMIZER['step_size']

        if isinstance(self.OPTIMIZER['gamma'], list):
            self.hp['gamma'] = tune.loguniform(self.OPTIMIZER['gamma'][0], self.OPTIMIZER['gamma'][1])
            self.is_hp = True
        else:
            self.hp['gamma'] = self.OPTIMIZER['gamma']
        """

    def hold_out_split(self, batch_size):
        
        trainloader = DataLoader(self.trainset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=1)
        
        testloader = DataLoader(self.testset, 
                                batch_size=batch_size,
                                num_workers=1)

        return trainloader, testloader


    def k_fold_split(self, batch_size):
        kfold = KFold(n_splits=self.VALIDATION['folds'], shuffle=True)
        _, testloader = self.hold_out_split(batch_size)
        trainloader_list, valloader_list = [], []
        for (train_ids, test_ids) in kfold.split(self.trainset):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader_list.append(torch.utils.data.DataLoader(
                            self.trainset, 
                            batch_size=batch_size,
                            sampler=train_subsampler,
                            num_workers=1))

            valloader_list.append(torch.utils.data.DataLoader(
                            self.trainset,
                            batch_size=batch_size,
                            sampler=val_subsampler,
                            num_workers=1))
    
        return trainloader_list, valloader_list, testloader


    def preproc_data(self):
        transform = transforms.Resize((
                        self.DATA['img-res'][0], 
                        self.DATA['img-res'][1]))

        self.dataset = ImageDataset(
                        data_dir=self.DATA['location'],
                        channels=self.channels,
                        transform=transform)

        test_size = int(self.VALIDATION['test_size'] * len(self.dataset))
        train_size = len(self.dataset) - test_size
        self.trainset, self.testset = torch.utils.data.random_split(self.dataset, 
                        [train_size, test_size])

        torch.save(self.dataset, './data/raw/dataset.pt')
        torch.save(self.trainset, './data/raw/trainset.pt')
        torch.save(self.testset, './data/raw/testset.pt')

        # TODO: create and upload artifacts both in S3 and wandb

    def train_trial(self, hp, checkpoint_dir=None):

        os.chdir(TUNE_ORIG_WORKING_DIR)

        loss_func = nn.NLLLoss()

        if self.VALIDATION['folds']:

            callbacks = [
                TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end"),
                EarlyStopping(monitor="ptl/loss")
            ]

            trainer = Trainer(fast_dev_run=False, 
                            max_epochs=hp['max_epochs'], 
                            callbacks=callbacks,
                            log_every_n_steps=1,
                            strategy="ddp", 
                            enable_progress_bar=False)
        
            trainloader, valloader, _ = self.k_fold_split(batch_size=hp['batch_size'])
            for (fold_idx, fold) in enumerate(trainloader):
                net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)

                trainer.fit(net, trainloader[fold_idx], valloader[fold_idx])
                    
        else:

            callbacks = [
                TuneReportCallback({"loss": "ptl/loss"}, on="fit_end"),
                EarlyStopping(monitor="ptl/loss")
            
            ]

            trainer = Trainer(fast_dev_run=False, 
                            max_epochs=hp['max_epochs'], 
                            logger=wandb_logger, 
                            callbacks=callbacks,
                            log_every_n_steps=1,
                            strategy="ddp", 
                            enable_progress_bar=False)

            trainloader, _ = self.hold_out_split(batch_size=hp['batch_size'])
            net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)
            
            trainer.fit(net, trainloader)


    def train_opt(self, hp):

        wandb_logger = WandbLogger(project=self.PROJECT['name'], name=self.PROJECT['experiment'])
        loss_func = nn.NLLLoss()

        callbacks = [
            ModelCheckpoint(dirpath="./models/", monitor='ptl/loss', save_top_k=3, save_last=True),
            EarlyStopping(monitor="ptl/loss")
        ]
        
        self.trainer = Trainer(fast_dev_run=False, 
                            max_epochs=hp['max_epochs'],
                            logger=wandb_logger,
                            callbacks=callbacks,
                            log_every_n_steps=1,
                            strategy="ddp")
    
        self.trainloader, self.testloader = self.hold_out_split(batch_size=hp['batch_size'])
        self.net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)
        
        self.trainer.fit(self.net, self.trainloader)


    def train(self):
        if self.is_hp:

            trainable = tune.with_parameters(self.train_trial, checkpoint_dir=None)

            scheduler = ASHAScheduler(
                max_t=self.TRAINING_HP['max_epochs'],
                grace_period=1,
                reduction_factor=2)

            #reporter = CLIReporter(
            #    parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
            #    metric_columns=["loss", "mean_accuracy", "training_iteration"])

            analysis = tune.run(
                trainable,
                resources_per_trial={
                    "cpu": 1,
                    "gpu": 0
                },
                local_dir=".",
                metric="loss",
                mode="min",
                config=self.hp,
                callbacks=[WandbLoggerCallback(
                    project=self.PROJECT['name']+'-HP',
                    group=self.PROJECT['experiment'],
                    api_key=self.WANDB['key'],
                    log_config=False)],
                num_samples=self.TUNING['number_trials'],
                scheduler=scheduler,
                name="tune_hp",
                verbose=0)

            final_config = analysis.best_config
            
            self.train_opt(final_config)

        else:
            self.train_opt(self.hp)


    def eval(self):
        if len(self.testloader) > 0:
            self.trainer.test(self.net, self.testloader)
            
            #wandb.log({"Accuracy Test Set": (100*correct/total).item()})
        else:
            print('Test set is empty. Step skipped.')