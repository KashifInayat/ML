from __future__ import print_function

import shutil
import threading

import torch

from torch import nn
from functools import partial

import numpy as np
import math

import sys
import os
import matplotlib.pyplot as plt

class PropagationMonitor(object):

    # Note that update, or batch, starts from 0, whereas epoch starts from 1.
    def __init__(self, module_types, mode="train", max_update=-1, max_epoch=0):
        self.handles = []
        self.max_update = max_update
        self.max_epoch = max_epoch
        
        self.mode = mode
        self.update = 0
        self.epoch = -1
        if module_types is not None:
            self.module_types = {}
            for x in module_types:
                self.module_types[x] = None
        else:
            self.module_types = None
        
    def set_network(self, net):
        self.net = net

    def start_update(self, mode,  epoch, update):
        # print('********************************* Start update, mode, epoch, batch index:', mode, epoch, update)
        if self.mode!=mode:
            return
        # print('max_update, update: ', self.max_update, update)  
        self.update = update
        if self.max_update==-1: # do it in start_epoch
            return
        if epoch>self.max_epoch and self.max_epoch!=-1:
            return

        if len(self.handles)==0 and update<=self.max_update:
            self.register_hooks()

        if update<=self.max_update:
            # print(' update more max_update, update: ', self.max_update, update)
            self.start_update_more(mode, epoch, update)

    def end_update(self, mode, epoch, update):
        # print('End update')
        if self.mode!=mode:
            return

        if self.max_update==-1: # do it in start_epoch
            # print('self.max_update', self.max_update)
            return
        if epoch>self.max_epoch and self.max_epoch!=-1:
            return

        if len(self.handles)!=0 and update==self.max_update:
            self.unregister_hooks() # only one at the beginning and the end

        if update<=self.max_update:
            self.end_update_more(mode, epoch, update) # call in the region

    def start_epoch(self, mode, epoch):
        if self.mode!=mode:
            return

        self.epoch = epoch
        if len(self.handles)==0 and (self.max_epoch==-1 or
            epoch<=self.max_epoch) and self.max_update==-1:
            self.register_hooks()  # we postpone this to "update" functions

        if self.max_epoch==-1 or epoch<self.max_epoch:
            self.start_epoch_more(mode, epoch) # this is not postponed

    def end_epoch(self, mode, epoch):
        if self.mode!=mode:
            return

        if len(self.handles)!=0 and self.max_update==-1:
            self.unregister_hooks()
        # we unregister hooks at the end of every epoch
        # otherwise, we cannot save the model. 

        if self.max_epoch==-1 or epoch<=self.max_epoch:
            self.end_epoch_more(mode, epoch)


    def register_hooks(self):
        for n, m in self.net.named_modules():
            if m==self.net:
                continue
            if (self.module_types is None) or type(m) in self.module_types:
                m.name = n
                handle = m.register_forward_hook(
                            self.monitor_activations)
                self.handles.append(handle)
                handle = m.register_backward_hook(
                            self.monitor_gradients)
                self.handles.append(handle)

    def unregister_hooks(self):
        for h in self.handles:
            h.remove()
        del self.handles[:]
    

    def start_epoch_more(self, mode, epoch):
        pass

    def end_epoch_more(self, mode, epoch):
        pass

    def start_update_more(self, mode, epoch, update):
        pass

    def end_update_more(self, mode, epoch, update):
        pass

    # Note that due to DataParallel, this function can be called multiple times
    # for each module in an iteration
    def monitor_activations(self, module, input, output):
        pass

    # Note that due to DataParallel, this function can be called multiple times
    # for each module in an iteration
    def monitor_gradients(self, module, input, output):
        pass


# SingleBatchStatisticsPrinter is an important class to debug the training process.
# We should maintain this class actively.
# This class has been validated for multi-gpu.
# Unlike HistoWriter, this doesn't write out gradients for parameters. 
# TODO: Add param grad support

class SingleBatchStatisticsPrinter(PropagationMonitor):

    def __init__(self, module_types, mode="train", max_update=-1, max_epoch=-1,
        save=False, num_samples_to_save=128, num_features_to_save=1):
        super(SingleBatchStatisticsPrinter, self).__init__(module_types, mode=mode,
                max_update=max_update, max_epoch=max_epoch)

        self.tensors = {}
        self.save = save
        self.num_samples_to_save = num_samples_to_save
        self.num_features_to_save = num_features_to_save
        if save:
            if not os.path.exists("./stat_logs"):
                os.mkdir("./stat_logs")
            else:
                shutil.rmtree("./stat_logs")
                os.mkdir("./stat_logs")

        self.lock = threading.Lock()

    def start_epoch_more(self, mode, epoch):
        self.tensors = {}

    def end_epoch_more(self, mode, epoch):
        tensors_to_save = {}
        for k,v in self.tensors.items():
            if v.dim()==1: # this is just a wrapper module
                continue
            n = min(self.num_samples_to_save, v.shape[0])
            m = min(self.num_features_to_save, v.shape[1])
            tensors_to_save[k] = v[:n,:m].clone()

        path = "./stat_logs/tensors-%s-ep%d.pt" % (mode, epoch)
        torch.save(tensors_to_save, path)

    def monitor_activations(self, module, input, output):
        self.lock.acquire(True)
        self.tensors["actin#"+module.name] = None
        self.lock.release()

	# the `input` of hook function is always a tuple of ONE elements.
        input = input[0]
        self.tensors["actin#"+module.name] = input.cpu()
        self.tensors["actout#"+module.name] = output.cpu()

        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # calculate n, the number of input connection
            if isinstance(module, nn.Linear):
                n = module.weight.data.shape[1]
            else:
                s = module.weight.shape
                n = s[1]*s[2]*s[3]            
            
            var = input.var()
            e2 = (input**2).mean()
            print('[activations/inputs]: %s ~ Variance:%f, Ex^2:%f (in_connections:%d) ->  [output] ~ Variance: %f' % (module.name, var, e2, n, output.data[:,:].var()))
        '''
        else:
            var_in = input[0].data.var()
            var_out = output.data.var()
            gain = var_out/var_in
            print('activations: %s %f  ->  %f (gain:%f)' %(module.name, var_in, var_out, gain))

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                for ch in range(1):
                    var_in = input[0].data[:,ch].var()
                    var_out = output.data[:,ch].var()
                    gain = var_out/var_in
                    print('\t\t %s %f  ->  %f (gain:%f) (ch%d)' %(module.name, var_in, var_out, gain,ch))
        '''


    def monitor_gradients(self, module, inputa, output):
        assert len(output)==1
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            # p,p,g
            input_idx = 0 
        elif isinstance(module, nn.Linear):
            # b,g,w
            if module.bias is not None:
                input_idx = 1 
            else:
                input_idx = 0
        else: ## relu is 0
            # g
            input_idx = 0

        if module.name=="":
            return
        #print ("module.name: ", module.name, "n: ", inputa[-1].shape )
        ni = inputa[-1].shape[1]
        ni_1 = inputa[input_idx].shape[1]
       
        #print (n, n_1)
        self.lock.acquire(True)
        self.tensors["gradin#"+module.name] = None
        self.lock.release()

        self.tensors["gradin#"+module.name] = output[0].cpu()
        if inputa[input_idx] is not None:
            self.tensors["gradout#"+module.name] = inputa[input_idx].cpu()
            #self.tensors["gradout#"+module.name] = inputa[input_idx].cpu() * 1/ (ni**0.5)
            #self.tensors["gradout#"+module.name] = inputa[input_idx].cpu() * 1/ ((ni*ni_1)**0.5)
            #self.tensors["gradout#"+module.name] = inputa[input_idx].cpu() * 1/ ( ((ni*ni_1)**0.5)* module.weight.std())
            #self.tensors["gradout#"+module.name] = inputa[input_idx].cpu() * 1/ ( ((ni)**0.5)* module.weight.std())
        var_in = output[0].var()
        if inputa[input_idx] is None:
            print('[gradients]: %s %f  ->  not computed' %(module.name, var_in))
        else:
            var_out = inputa[input_idx].var()
            gain = var_out/var_in
            print('[gradients]: %s Variance In: %f  -> Variance Out: %f (Gain: %f)' %(module.name, var_in, var_out, gain))
        print ("std(Δw') / std(w): =",self.tensors["gradout#" + module.name].std(),"/",module.weight.std(),"=", self.tensors["gradout#" + module.name].std() / module.weight.std())
        print ("Weight magutide:", torch.norm(inputa[input_idx].cpu()))
		#colors = ['red', 'tan', 'lime']
        plt.hist(inputa[input_idx].cpu().flatten().numpy(), density=True, bins=500, color='green', label='Gradient Δw ('+module.name+')', alpha=0.5)
        plt.hist(self.tensors["gradout#"+module.name].flatten().numpy(), density=True, bins=500, color='blue', label='Gradient Δw\' ('+module.name+')', alpha=0.5)
        plt.legend()
        plt.savefig("Fig1-"+module.name[5]+".eps")
        plt.clf()
        plt.cla()
        plt.close()

        '''
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            for ch in range(1):
                var_in = output[0][:,ch].data.var()
                var_out = inputa[input_idx][:,ch].data.var()
                gain = var_out/var_in
                print('\t\t %s %f  ->  %f (gain: %f) (ch%d)' %(module.name, var_in, var_out, gain, ch))
        '''
       

