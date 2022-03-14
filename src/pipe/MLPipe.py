import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_lightning import Trainer
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune

from sklearn.model_selection import KFold

import wandb

import uuid
import os

from src.data.make_dataset import ImageDataset
from src.models.model import Net

TUNE_ORIG_WORKING_DIR = os.getcwd()


class MLPipe():


    def __init__(self, config_dict=None):
        
        self.__parse_config_dict__(config_dict)

        self.__set_hp_params__()

        #wandb.login(key=self.WANDB_KEY['key'])
        #wandb.init(project=self.PROJECT['name'], name=self.PROJECT['experiment'])


    def __parse_config_dict__(self, config_dict):
        # TODO: do actual parsing and validate argument, else throw error

        self.DATA = config_dict['data']
        self.TRAINING_HP = config_dict['training']
        self.OPTIMIZER = config_dict['optimizer']
        self.WANDB_KEY = config_dict['wandb']
        self.PROJECT = config_dict['project']
        self.VALIDATION = config_dict['validation']

        try:
            self.PROJECT['experiment']
        except KeyError:
            self.PROJECT['experiment'] = uuid.uuid4().hex

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


    def hold_out_split(self, batch_size):
        test_size = int(self.VALIDATION['test_size'] * len(self.dataset))
        train_size = len(self.dataset) - test_size
        trainset, testset = torch.utils.data.random_split(self.dataset, 
                        [train_size, test_size])

        trainloader = DataLoader(trainset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=4)
        
        testloader = DataLoader(testset, 
                                batch_size=batch_size,
                                num_workers=4)

        return trainset, testset, trainloader, testloader


    def k_fold_split(self, batch_size):
        kfold = KFold(n_splits=self.VALIDATION['folds'], shuffle=False)
        trainset, _, _, testloader = self.hold_out_split(batch_size=batch_size)
        trainloader_list, valloader_list = [], []
        for (train_ids, test_ids) in kfold.split(trainset):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader_list.append(torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=batch_size,
                            sampler=train_subsampler,
                            num_workers=4))

            valloader_list.append(torch.utils.data.DataLoader(
                            trainset,
                            batch_size=batch_size,
                            sampler=val_subsampler,
                            num_workers=4))
    
        return trainloader_list, valloader_list, testloader


    def preproc_data(self):
        transform = transforms.Resize((
                        self.DATA['img-res'][0], 
                        self.DATA['img-res'][1]))

        self.dataset = ImageDataset(
                        data_dir=self.DATA['location'],
                        channels=self.channels,
                        transform=transform)


    def train_trial(self, hp):

        os.chdir(TUNE_ORIG_WORKING_DIR)

        loss_func = nn.CrossEntropyLoss()

        metrics = {"loss": "ptl/val_loss"}
        callbacks = [TuneReportCallback(metrics, on="validation_end")]
        trainer = Trainer(fast_dev_run=False, max_epochs=hp['max_epochs'], callbacks=callbacks, strategy="ddp")

        if self.VALIDATION['folds']:
            trainloader, valloader, testloader = self.k_fold_split(batch_size=hp['batch_size'])
            for (fold_idx, fold) in enumerate(trainloader):
                net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)

                trainer.fit(net, trainloader[fold_idx], valloader[fold_idx])
                    
        else:
            _, _, trainloader, testloader = self.hold_out_split(batch_size=hp['batch_size'])
            net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)
            
            trainer.fit(net, trainloader)


    def train_opt(self, hp):

        loss_func = nn.CrossEntropyLoss()

        #metrics = {"loss": "ptl/val_loss"}
        #callbacks = [TuneReportCallback(metrics, on="validation_end")]
        self.trainer = Trainer(fast_dev_run=False, max_epochs=hp['max_epochs'], strategy="ddp")#, callbacks=callbacks)

        if self.VALIDATION['folds']:
            self.trainloader, self.valloader, self.testloader = self.k_fold_split(batch_size=hp['batch_size'])
            for (fold_idx, fold) in enumerate(self.trainloader):
                self.net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)

                self.trainer.fit(self.net, self.trainloader[fold_idx], self.valloader[fold_idx])
                    
        else:
            _, _, self.trainloader, self.testloader = self.hold_out_split(batch_size=hp['batch_size'])
            self.net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)
            
            self.trainer.fit(self.net, self.trainloader)


    def train(self):
        if self.is_hp:
            trainable = tune.with_parameters(self.train_trial)

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
                num_samples=self.OPTIMIZER['number_trials'],
                scheduler=scheduler,
                name="tune_hp",
                verbose=1)

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