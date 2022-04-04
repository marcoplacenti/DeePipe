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

import yaml
import yamale

import uuid
import datetime
import os

from src.data.make_dataset import ImageDataset
from src.models.model import Net
from src.pipe.AWSConnector import AWSConnector

TUNE_ORIG_WORKING_DIR = os.getcwd()
os.environ['WANDB_SILENT']="true"


class MLPipe():

    def __init__(self, config_file=None, name=None, experiment=None, task=None):
        
        if not config_file:
            self.config_file_flag = False
            assert name
            assert task
            config_dict = {'project': {'name': name, 'task': task}}
            if not experiment:
                experiment = str(datetime.datetime.now())+'-'+str(uuid.uuid4().hex)
            config_dict['project']['experiment'] = experiment
        else:
            self.config_file_flag = True
            config_dict = self.__validate_configuration__(config_file)
            self.__parse_config_dict__(config_dict)

        self.__set_aws_connector__()
        self.__setup_dirs__()        

    
    def __validate_configuration__(self, config_file):
        schema = yamale.make_schema('./config/schema.yml')
        data = yamale.make_data(config_file)

        try:
            yamale.validate(schema, data)
            with open(config_file) as infile:
                config_dict = yaml.load(infile, Loader=yaml.SafeLoader)

            return config_dict
        except Exception as e:
            print(e)
            print("Provided yaml file not acceptable.")
            exit(1)


    def __set_aws_connector__(self):
        self.aws_connector = AWSConnector(self.PROJECT['name'])


    def __parse_config_dict__(self, config_dict):

        self.DATA = config_dict['data']
        self.TRAINING_HP = config_dict['training']
        self.OPTIMIZATION = config_dict['optimization']
        self.TUNING = config_dict['tuning']
        self.PROJECT = config_dict['project']
        self.VALIDATION = config_dict['validation']

        try:
            self.PROJECT['experiment']
        except KeyError:
            self.PROJECT['experiment'] = str(datetime.datetime.now())+'-'+str(uuid.uuid4().hex)

        self.channels = (1 if self.DATA['greyscale'] else 3)

    def __fill_hp_dict(self, max_epochs, batch_size, optimizer, learning_rate):
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


    def __set_hp_params__(self, max_epochs=None, batch_size=None, optimizer=None, learning_rate=None):
        self.is_hp = False
        if self.config_file_flag:
            self.__fill_hp_dict(self.TRAINING_HP['max_epochs'], 
                                self.TRAINING_HP['batch_size'], 
                                self.OPTIMIZATION['optimizer'], 
                                self.OPTIMIZATION['learning_rate'])
            
        else:
            self.__fill_hp_dict(max_epochs, batch_size, optimizer, learning_rate)



    def __setup_dirs__(self):
        if not os.path.exists('./data'):
            os.makedirs('./data')
        if not os.path.exists('./data/raw'):
            os.makedirs('./data/raw')
        if not os.path.exists('./data/raw/'+self.DATA['location'].split('/')[2]):
            os.makedirs('./data/raw/'+self.DATA['location'].split('/')[2])

        if not os.path.exists('./trials'):
            os.makedirs('./trials')
        if not os.path.exists('./trials/'+self.PROJECT['experiment']):
            os.makedirs('./trials/'+self.PROJECT['experiment'])


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

        self.upload_artifacts('data', './data/raw/'+self.DATA['location'].split('/')[2])

    def upload_artifacts(self, artifact_type, path):
        session, bucket = self.aws_connector.S3_session()
        s3_client = session.client('s3')

        run = wandb.init(project=self.PROJECT['name'], name=self.PROJECT['experiment'], entity='dma')

        for root, _, files in os.walk(path):
            for obj in files:
                if artifact_type == 'data':
                    obj_name = self.PROJECT['name']+'/'+artifact_type+'/'+obj
                elif artifact_type == 'trials':
                    obj_name = self.PROJECT['name']+'/'+artifact_type+'/'+'/'.join(root.split('/')[2:])+'/'+obj
                s3_client.upload_file(os.path.join(root, obj), bucket, obj_name)

                artifact = wandb.Artifact(obj, type=artifact_type)
                artifact.add_reference('s3://'+bucket+'/'+obj_name)
                run.log_artifact(artifact)

        del s3_client
        del session
        wandb.finish()


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
                            log_every_n_steps=1)
    
        self.trainloader, self.testloader = self.hold_out_split(batch_size=hp['batch_size'])
        self.net = Net(dataset=self.dataset, in_channels=self.channels, hp=hp, loss_func=loss_func)
        
        self.trainer.fit(self.net, self.trainloader)


    def train(self, max_epochs=None, batch_size=None, optimizer=None, learning_rate=None):
        self.__set_hp_params__(max_epochs, batch_size, optimizer, learning_rate)
        if self.is_hp:

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

            self.upload_artifacts('trials', './trials/'+self.PROJECT['experiment'])

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