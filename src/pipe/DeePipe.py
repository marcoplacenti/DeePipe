import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
import ray

from sklearn.model_selection import KFold

import wandb

import uuid
import datetime
import os
import tarfile

import logging

from src.data.make_dataset import ImageDataset
from src.models.model import Model
from src.pipe.AWSConnector import AWSConnector
from src.pipe.Configurator import Configurator
from src.models.architectures import *


TUNE_ORIG_WORKING_DIR = os.getcwd()
os.environ['WANDB_SILENT']="true"

logging.getLogger().setLevel(logging.INFO)

class DeePipe():

    def __init__(self, config_file=None, name=None, experiment=None, task=None):

        if not config_file:
            self.config_file_flag = False
            assert name
            assert task
            self.PROJECT = {'name': name, 'task': task}
            if not experiment:
                experiment = str(datetime.datetime.now())+'-'+str(uuid.uuid4().hex)
            self.PROJECT['experiment'] = experiment
            self.__set_aws_connector__()   
        else:
            self.config_file_flag = True
            self.configurator = Configurator(config_file)
            self.configurator.validate()
            config_dict = self.configurator.get_dict()
            self.__parse_config_dict__(config_dict)
            self.__set_hp_params__()
            self.__set_aws_connector__()

            if self.configurator.get_data_flag():
                self.preproc_data()
            if self.configurator.get_train_flag():
                self.train()
            if self.configurator.get_evaluation_flag():
                self.eval()
            if self.configurator.get_deployment_flag():
                self.deploy()
            

    def __set_aws_connector__(self):
        self.aws_connector = AWSConnector(self.PROJECT['name'])

    def __parse_config_dict__(self, config_dict):

        self.DATA = config_dict['data']
        self.MODEL_ARCHITECTURE = config_dict['model_architecture']
        self.TRAINING_HP = config_dict['training']
        self.OPTIMIZATION = config_dict['optimization']
        self.TUNING = config_dict['tuning']
        self.PROJECT = config_dict['project']
        self.VALIDATION = config_dict['validation']
        self.DEPLOYMENT = config_dict['deployment']

        try:
            self.PROJECT['experiment']
        except KeyError:
            self.PROJECT['experiment'] = str(datetime.datetime.now())+'-'+str(uuid.uuid4().hex)

        self.channels = (1 if self.DATA['greyscale'] else 3)

    def __fill_hp_dict(self, max_epochs, batch_size, optimizer, learning_rate):
        self.is_hp = False
        self.hp = {'max_epochs': max_epochs}

        if isinstance(batch_size, list):
            self.hp['batch_size'] = tune.choice(batch_size)
            self.is_hp = True
        else:
            self.hp['batch_size'] = batch_size

        if isinstance(learning_rate, list):
            self.hp['lr'] = tune.loguniform(learning_rate[0], learning_rate[1])
            self.is_hp = True
        else:
            self.hp['lr'] = learning_rate

        if isinstance(optimizer, list):
            self.hp['optimizer'] = tune.choice(optimizer)
            self.is_hp = True
        else:
            self.hp['optimizer'] = optimizer


    def __set_hp_params__(self, model=None, max_epochs=None, batch_size=None, optimizer=None, learning_rate=None, loss_function=None, number_trials=None):
        if not self.config_file_flag:
            self.TRAINING_HP = {'max_epochs': max_epochs, 'batch_size': batch_size}
            self.OPTIMIZATION = {'optimizer': optimizer, 'learning_rate': learning_rate, 'loss_fnc': loss_function}
            self.TUNING = {'number_trials': number_trials}
            self.MODEL_ARCHITECTURE = {'name': model}

        self.__fill_hp_dict(self.TRAINING_HP['max_epochs'], 
                                self.TRAINING_HP['batch_size'], 
                                self.OPTIMIZATION['optimizer'], 
                                self.OPTIMIZATION['learning_rate'])


    def __setup_data_dirs__(self):
        if not os.path.exists('./data'):
            os.makedirs('./data')
        if not os.path.exists('./data/raw'):
            os.makedirs('./data/raw')
        if not os.path.exists('./data/raw/'+self.DATA['location'].split('/')[2]):
            os.makedirs('./data/raw/'+self.DATA['location'].split('/')[2])


    def __setup_train_dirs__(self):
        if not os.path.exists('./trials'):
            os.makedirs('./trials')
        if not os.path.exists('./trials/'+self.PROJECT['experiment']):
            os.makedirs('./trials/'+self.PROJECT['experiment'])


    def __set_data_params__(self, location, img_res, greyscale, test_size, folds):
        self.DATA = {'location': location, 'img-res': img_res, 'greyscale': greyscale}
        self.VALIDATION = {'test_size': test_size, 'folds': folds}

        self.channels = (1 if self.DATA['greyscale'] else 3)


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

    # TODO: enable downloading when data location is s3
    def preproc_data(self, location=None, img_res=None, greyscale=None, test_size=None, folds=None):
        if not self.config_file_flag:
            self.__set_data_params__(location, img_res, greyscale, test_size, folds)

        self.__setup_data_dirs__()

        transform = transforms.Resize((
                        self.DATA['img-res'][0], 
                        self.DATA['img-res'][1]))

        self.dataset = ImageDataset(
                        data_dir=self.DATA['location'],
                        channels=self.channels,
                        transform=transform)

        torch.save(self.dataset, './data/raw/'+self.DATA['location'].split('/')[2]+'/dataset.pt')

        if set(['training', 'testing']).issubset(os.listdir(self.DATA['location'])):
            self.trainset = ImageDataset(
                        data_dir=self.DATA['location']+'training/',
                        channels=self.channels,
                        transform=transform
            )

            self.testset = ImageDataset(
                        data_dir=self.DATA['location']+'testing/',
                        channels=self.channels,
                        transform=transform
            )

        else: 
            test_size = int(self.VALIDATION['test_size'] * len(self.dataset))
            train_size = len(self.dataset) - test_size
            self.trainset, self.testset = torch.utils.data.random_split(self.dataset, 
                        [train_size, test_size])

        torch.save(self.trainset, './data/raw/'+self.DATA['location'].split('/')[2]+'/trainset.pt')
        torch.save(self.testset, './data/raw/'+self.DATA['location'].split('/')[2]+'/testset.pt')

        self.aws_connector.upload_artifacts('data', './data/raw/'+self.DATA['location'].split('/')[2], self.PROJECT)


    def train_trial(self, hp, checkpoint_dir=None):

        os.chdir(TUNE_ORIG_WORKING_DIR)

        wandb_logger = WandbLogger(project=self.PROJECT['name'], name=self.PROJECT['experiment'], entity='dma')
        loss_func = eval('nn.'+self.OPTIMIZATION['loss_fnc'])()

        if self.VALIDATION['folds']:

            callbacks = [
                TuneReportCallback({"loss": "ptl/val_loss"}, on="validation_end"),
                EarlyStopping(monitor="ptl/loss")
            ]

            trainer = Trainer(fast_dev_run=False, 
                            max_epochs=hp['max_epochs'], 
                            callbacks=callbacks,
                            log_every_n_steps=1,
                            enable_progress_bar=False)
        
            trainloader, valloader, _ = self.k_fold_split(batch_size=hp['batch_size'])
            for (fold_idx, fold) in enumerate(trainloader):
                net = Model()
                net.set_architecture(self.model)
                net.set_hyperparameters(hp['optimizer'], hp['lr'], loss_func)

                trainer.fit(net, fold, valloader[fold_idx])
                    
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
                            enable_progress_bar=False)

            trainloader, _ = self.hold_out_split(batch_size=hp['batch_size'])
            net = Model()
            net.set_architecture(self.model)
            net.set_hyperparameters(hp['optimizer'], hp['lr'], loss_func)
            
            trainer.fit(net, trainloader)


    def train_opt(self, hp):

        wandb_logger = WandbLogger(project=self.PROJECT['name'], name=self.PROJECT['experiment'], entity='dma')
        loss_func = eval('nn.'+self.OPTIMIZATION['loss_fnc'])()

        checkpoint_callback = ModelCheckpoint(dirpath="./tmp/models/checkpoints", 
                                filename='{epoch}-{loss:.2f}', 
                                monitor='ptl/loss', 
                                save_top_k=3, 
                                save_last=True)
        checkpoint_callback.FILE_EXTENSION = '.pth'

        callbacks = [
            checkpoint_callback,
            EarlyStopping(monitor="ptl/loss")
        ]

        self.trainer = Trainer(fast_dev_run=False, 
                            max_epochs=hp['max_epochs'],
                            logger=wandb_logger,
                            callbacks=callbacks,
                            log_every_n_steps=1)
    
        self.trainloader, self.testloader = self.hold_out_split(batch_size=hp['batch_size'])
        self.net = Model()
        self.net.set_architecture(self.model)
        self.net.set_hyperparameters(hp['optimizer'], hp['lr'], loss_func)
        
        self.trainer.fit(self.net, self.trainloader)

        model_scripted = self.net.to_torchscript()#torch.jit.script(self.net) # Export to TorchScript
        torch.jit.save(model_scripted, './tmp/models/final.pth') # Save

    def train(self, model=None, max_epochs=None, batch_size=None, optimizer=None, learning_rate=None, number_trials=None):
        if not self.config_file_flag:
            self.__set_hp_params__(model, max_epochs, batch_size, optimizer, learning_rate, number_trials)
        self.__setup_train_dirs__()

        self.model = eval(self.MODEL_ARCHITECTURE['name'])(self.channels, self.dataset.get_num_classes())

        if self.is_hp:
            logging.info("Running HP tuner...")
            ray.shutdown()
            ray.init(log_to_driver=False)
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
                local_dir='./trials',
                metric="loss",
                mode="min",
                config=self.hp,
                callbacks=[WandbLoggerCallback(
                    project=self.PROJECT['name']+'-HP',
                    group=self.PROJECT['experiment'],
                    api_key=os.environ['WANDB_KEY'],
                    log_config=False)],
                num_samples=self.TUNING['number_trials'],
                scheduler=scheduler,
                name=self.PROJECT['experiment'],
                verbose=0)

            self.aws_connector.upload_artifacts('trials', './trials/'+self.PROJECT['experiment'], self.PROJECT)

            final_config = analysis.best_config
            
            self.train_opt(final_config)
            
        else:
            self.train_opt(self.hp)

        model_checkpoints = os.listdir('./tmp/models/checkpoints')
        for ckpt in model_checkpoints:
            if '=' in ckpt:
                os.rename('./tmp/models/checkpoints/'+ckpt, './tmp/models/checkpoints/'+ckpt.replace('=', '_'))
        
        self.aws_connector.upload_artifacts('models', './tmp/models/checkpoints', self.PROJECT)
        self.aws_connector.upload_artifacts('models', './tmp/models/final/', self.PROJECT)


    def eval(self):
        if len(self.testloader) > 0:
            self.trainer.test(self.net, self.testloader)
            
            #wandb.log({"Accuracy Test Set": (100*correct/total).item()})
        else:
            logging.info("Test set is empty. Step skipped.")

    # TODO: only deploy when test requirements are met in config file - allow tarfile as args
    def deploy(self, tarfile_name=None, inference_dir=None, model_dir=None):
        tarfile_name = './tmp/models/myModel.tar.gz'
        with tarfile.open(tarfile_name, "w:gz") as tar:
            tar.add('./src/pipe/inference/', arcname='.')
            tar.add('./src/models/', arcname='./src/models/')
        self.aws_connector.deploy(tarfile_name, self.DEPLOYMENT['endpoint_name'], self.DEPLOYMENT['instance_type'])