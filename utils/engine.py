'''
Author: Wojciech Fedorko
Collaborators: Julian Ding, Abhishek Kajal
'''

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import time
import numpy as np

# PyTorch imports
from torch import device, load, save
from torch.nn import DataParallel
from torch.cuda import is_available

from utils.logging_utils import CSVData



class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model, dataset, config):
        self.model = model
        
        
        # Determine the device to be used for model training and inference
        if (config.device == 'gpu') and config.gpu_list:
            print("Requesting GPUs. GPU list : " + str(config.gpu_list))
            self.devids=["cuda:{0}".format(x) for x in config.gpu_list]
            print("Main GPU : " + self.devids[0])

            if is_available():
                self.device=device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model=DataParallel(self.model, device_ids=config.gpu_list, dim=0)
                print("CUDA is available")
            else:
                self.device=device("cpu")
                print("CUDA is not available")
        else:
            print("Unable to use GPU")
            self.device=device("cpu")

        # Send the model to the selected device
        self.model.to(self.device)

        if config.optimizer=='Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        elif config.optimizer=='SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)
        else:
            raise ValueError('select optimizer explicitly by providing config.optimizer field\n'\
                             'config.optimizer field is set to {}'.format(config.optimizer))
            
            
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None


        self.dset=dataset

        self.train_dldr=DataLoader(self.dset,
                                   batch_size=config.batch_size_train,
                                   shuffle=False,
                                   sampler=SubsetRandomSampler(self.dset.train_indices),
                                   num_workers=config.num_workers_train)
        
        self.val_dldr=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices),
                                 num_workers=config.num_workers_val)
        self.val_iter=iter(self.val_dldr)
        
        self.test_dldr=DataLoader(self.dset,
                                  batch_size=config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices),
                                  num_workers=config.num_workers_test)

        self.dirpath=config.dump_path + "/"+time.strftime("%Y%m%d_%H%M%S") + "/"
        
        self.len_train_dldr=len(self.train_dldr)
        self.len_val_dldr=len(self.val_dldr)
        self.len_test_dldr=len(self.test_dldr)

        print('length of train dataloader: {}'.format(self.len_train_dldr))
        print('length of validation dataloader: {}'.format(self.len_val_dldr))
        print('length of test dataloader: {}'.format(self.len_test_dldr))
        
        if config.lr_scheduler is not None:
            print('config.lr_scheduler is {}'.format(config.lr_scheduler))
            if config.lr_scheduler=='SGDR':
                print('SGDR learning schedule')
                self.lr_schedule_period=int(config.lr_initial_period*self.len_train_dldr)
                self.lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.lr_schedule_period)
                self.lr_schedule_batches_since_restart=0
            else:
                raise NotImplementedError('{} scheduler is not implemented'.format(config.lr_scheduler))
        
                
        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump: {}".format(self.dirpath))
            os.makedirs(self.dirpath,exist_ok=True)

        self.config=config
        
        # Save a copy of the config in the dump path
        f_config=open(self.dirpath+"/config_log.txt","w")
        f_config.write(str(vars(config)))


    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU
            # if using CPU this has no effect
            self.data = self.data.to(self.device)
            self.label = self.label.to(self.device)

            
            linear_model_out = self.model(self.data)
            # Training
            
            self.loss = self.criterion(linear_model_out,self.label)
            
            
            softmax    = self.softmax(linear_model_out).detach().cpu().numpy()
            prediction = torch.argmax(linear_model_out,dim=-1)
            accuracy   = (prediction == self.label).sum().item() / float(prediction.nelement())        
            prediction = prediction.cpu().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : self.loss.detach().cpu().item(),
                'accuracy'   : accuracy}

    def backward(self):
        self.optimizer.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.optimizer.step()
                                          
            
    # ========================================================================
    def train(self, epochs=3.0, report_interval=10, valid_interval=1000):
        # Based on WaTCHMaL workshop and W's code
        
        # Keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss=1.0e6
        
        
        # Prepare attributes for data logging
        self.train_log, self.val_log = CSVData(self.dirpath+"log_train.csv"), CSVData(self.dirpath+"log_val.csv")
        # Set neural net to training mode
        self.model.train()
        # Initialize epoch counter
        epoch = 0.
        # Initialize iteration counter
        self.iteration = 0
        # Training loop
        while ((int(epoch+0.5) < epochs) ):
            print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            j = 0
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_dldr):

                # once in a while run valiation
                # as a sanity check run validation before we start training
                if i%valid_interval == 0:
                    self.model.eval()

                    try:
                        val_data = next(self.val_iter)
                    except StopIteration:
                        print("starting over on the validation set")
                        self.val_iter=iter(self.val_dldr)
                        val_data = next(self.val_iter)
                        
                    # Data and label
                    self.data = val_data[0]
                    self.label = val_data[1]
                    
                    res = self.forward(False)
                    print('... Iteration %d ... Epoch %1.2f ... Validation Loss %1.3f ... Validation Accuracy %1.3f' % (self.iteration,epoch,res['loss'],res['accuracy']))
                    
                    
                    self.model.train()

                    self.save_state()
                    mark_best=0
                    if res['loss']<best_val_loss:
                        best_val_loss=res['loss']
                        print('best validation loss so far!: {}'.format(best_val_loss))
                        self.save_state(best=True)
                        mark_best=1

                    self.val_log.record(['iteration','epoch','accuracy','loss','saved_best'],[self.iteration,epoch,res['accuracy'],res['loss'],mark_best])
                    self.val_log.write()
                    self.val_log.flush()
                
                # Data and label
                self.data = data[0]
                self.label = data[1]
                
                
                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                # Call backward: backpropagate error and update weights
                self.backward()
                
                
                # Log/Report
    
                # Record the current performance on train set
                if self.config.lr_scheduler is not None:
                    self.train_log.record(['iteration','epoch','accuracy','loss','lr'],
                                          [self.iteration,epoch,res['accuracy'],res['loss'],self.lr_scheduler.get_lr()[0]])
                else:
                    self.train_log.record(['iteration','epoch','accuracy','loss'],
                                          [self.iteration,epoch,res['accuracy'],res['loss']])
                #self.train_log.record(['iteration','epoch','accuracy','loss'],
                #                      [self.iteration,epoch,res['accuracy'],res['loss']])
                self.train_log.write()
                self.train_log.flush()
                
                # once in a while, report
                if i==0 or i%report_interval == 0:
                    if self.config.lr_scheduler is not None:
                        print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f ... LR %.3e' % (self.iteration,epoch,res['loss'],res['accuracy'],self.lr_scheduler.get_lr()[0]))
                    else:
                        print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (self.iteration,epoch,res['loss'],res['accuracy']))
                    
                # Epoch update
                epoch += 1./len(self.train_dldr)
                self.iteration += 1
                
                #learning rate update if any
                if self.config.lr_scheduler=='SGDR':
                    if self.lr_schedule_batches_since_restart==self.lr_schedule_period:
                        print('***************************************************')
                        print('learning schedule restart!!')
                        self.lr_schedule_period=int(self.lr_schedule_period/self.config.lr_schedule_division)
                        print('upcoming cycle will be {} batches = {} epochs long'.format(self.lr_schedule_period,
                                                                                          self.lr_schedule_period*self.len_train_dldr))
                        self.lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.lr_schedule_period)
                        self.lr_schedule_batches_since_restart=0
                              
                    self.lr_scheduler.step()
                    self.lr_schedule_batches_since_restart+=1    
                
                        
                if epoch >= epochs:
                    break
                                 
                    
        self.val_log.close()
        self.train_log.close()
        #np.save(self.dirpath + "/optim_state_array.npy", np.array(optim_state_list))
    
    # ========================================================================

    # Function to test the model performance on the validation dataset
    def validate(self):
        """
        Test the trained model on the validation set.
        
        Parameters: None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
       
        
        # Variables to output at the end
        val_loss = 0.0
        val_acc = 0.0
        val_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes= [],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for it, val_data in enumerate(self.val_dldr):
                
                sys.stdout.write("val_iterations : " + str(val_iterations) + "\n")
                
                self.data, self.label = val_data[0:2]
                

                # Run the forward procedure and output the result
                result = self.forward(False)
                val_loss += result['loss']
                val_acc += result['accuracy']
                
                # Add item to priority queues if necessary
                
                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")
                
                # Add the local result to the final result
                labels.extend(self.label)
                predictions.extend(result['prediction'])
                softmaxes.extend(result["softmax"])
                
                val_iterations += 1
                
        print(val_iterations)

        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)
        
        np.save(self.dirpath + "labels.npy", np.array(labels))
        np.save(self.dirpath + "predictions.npy", np.array(predictions))
        np.save(self.dirpath + "softmax.npy", np.array(softmaxes))  
        
        
    # ========================================================================
    
            
    def save_state(self,best=False):
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST" if best else ""),
                                     ".pth")
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        if self.lr_scheduler is not None and self.config.lr_scheduler=='SGDR':
            torch.save({
                'global_step': self.iteration,
                'lr_schedule_batches_since_restart': self.lr_schedule_batches_since_restart,
                'lr_schedule_period': self.lr_schedule_period,
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.model.state_dict()
            }, filename)
        else:
            torch.save({
                'global_step': self.iteration,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.model.state_dict()
            }, filename)
        print('Saved checkpoint as:', filename)
        return filename

    def restore_state(self, weight_file):
        
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
            if self.lr_scheduler is not None:
                if self.config.lr_scheduler=='SGDR' and self.config.lr_scheduler_restore:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                else:
                    print("lr_scheduler is present but state not restored")
                    print("config.lr_scheduler: {}, config.lr_scheduler_restore: {}".format(self.config.lr_scheduler,self.config.lr_scheduler_restore))
            
        print('Restoration complete.')
            
