import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import KFold

from src.data.make_dataset import ImageDataset
from src.models.model import Net


class MLPipe():
    def __init__(self, config_dict=None):
        self.DATA = config_dict['data']
        self.TRAINING_HP = config_dict['training']
        self.OPTIMIZER = config_dict['optimizer']
        self.WANDB_KEY = config_dict['wandb']
        self.PROJECT = config_dict['project']
        self.VALIDATION = config_dict['validation']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def hold_out_split(self):
        test_size = int(self.VALIDATION['test_size'] * len(self.dataset))
        train_size = len(self.dataset) - test_size
        trainset, testset = torch.utils.data.random_split(self.dataset, 
                        [train_size, test_size])

        trainloader = DataLoader(trainset, 
                                batch_size=self.TRAINING_HP['batch_size'], 
                                shuffle=True)
        
        testloader = DataLoader(testset, 
                                batch_size=self.TRAINING_HP['batch_size'])

        return trainset, testset, trainloader, testloader

    def k_fold_split(self):
        kfold = KFold(n_splits=self.VALIDATION['folds'], shuffle=True)
        trainset, _, _, testloader = self.hold_out_split()
        trainloader_list, valloader_list = [], []
        for (train_ids, test_ids) in kfold.split(trainset):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader_list.append(torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=self.TRAINING_HP['batch_size'],
                            sampler=train_subsampler))

            valloader_list.append(torch.utils.data.DataLoader(
                            trainset,
                            batch_size=self.TRAINING_HP['batch_size'],
                            sampler=val_subsampler))
    
        return trainloader_list, valloader_list, testloader

    def preproc_data(self):
        transform = transforms.Resize((
                        self.DATA['img-res'][0], 
                        self.DATA['img-res'][1]))

        self.dataset = ImageDataset(
                        data_dir=self.DATA['location'],
                        transform=transform)   

    def epoch_loop(self, dataloader, optimizer, scheduler=None):
        for epoch in range(self.TRAINING_HP['epochs']):
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                if (batch_idx+1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{self.TRAINING_HP['epochs']}], " +
                        f"Step [{batch_idx+1}/{len(dataloader)}], " +
                        f"Training Loss: {round(loss.item(),3)}")
            if scheduler:
                    scheduler.step()


    def train(self):
        if not self.DATA['greyscale']:
            in_channels = 3
        else:
            in_channels = 1

        self.loss_function = nn.CrossEntropyLoss()

        if self.VALIDATION['folds']:
            results = {}
            self.trainloader, self.valloader, self.testloader = self.k_fold_split()
            for (fold_idx, fold) in enumerate(self.trainloader):
                self.net = Net(in_channels=in_channels, 
                    num_classes=self.dataset.get_num_classes()).to(self.device)

                optimizer = torch.optim.Adam(self.net.parameters(), 
                                    lr=self.OPTIMIZER['learning_rate'])
                scheduler = None
                if self.OPTIMIZER['step_size'] and self.OPTIMIZER['gamma']:
                    scheduler = StepLR(optimizer, step_size=self.OPTIMIZER['step_size'], gamma=self.OPTIMIZER['gamma'])

                self.epoch_loop(dataloader=fold, optimizer=optimizer, scheduler=scheduler)

                with torch.no_grad():
                    correct, total = 0, 0
                    for images, labels in self.valloader[fold_idx]:
                        images, labels = images.to(self.device), labels.to(self.device)
                        output = self.net(images)
                        _, predicted = torch.max(output,1)
                        correct += (predicted == labels).sum()
                        total += labels.size(0)

                fold_acc = round((100*correct/total).item(), 3)
                print(f'Accuracy for fold {fold_idx+1}: {fold_acc}%')
                print('--------------------------------')
                results[fold_idx] = fold_acc
                    
            print(f'K-FOLD CROSS VALIDATION RESULTS FOR {len(self.trainloader)} FOLDS')
            print('--------------------------------')
            sum = 0.0
            for key, value in results.items():
                print(f'Fold {key}: {value} %')
                sum += value
            print(f'Average: {sum/len(results.items())} %')
        else:
            _, _, self.trainloader, self.testloader = self.hold_out_split()
            self.net = Net(in_channels=in_channels, 
                    num_classes=self.dataset.get_num_classes()).to(self.device)
            optimizer = torch.optim.Adam(self.net.parameters(), 
                                lr=self.OPTIMIZER['learning_rate'])
            scheduler = StepLR(optimizer, step_size=self.OPTIMIZER['step_size'], gamma=self.OPTIMIZER['gamma'])
            self.epoch_loop(dataloader=self.trainloader, optimizer=optimizer, scheduler=scheduler)
            

    def eval(self):
        if len(self.testloader) > 0:
            with torch.no_grad():
                correct, total = 0, 0
                for images, labels in self.testloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.net(images)
                    _, predicted = torch.max(output,1)
                    correct += (predicted == labels).sum()
                    total += labels.size(0)

            print('Accuracy of the model on test set: %.3f %%' %((100*correct/total).item()))
        else:
            print('Test set is empty. Step skipped.')