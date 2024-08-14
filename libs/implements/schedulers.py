import torch
import math
import warnings
import inspect
from collections import Counter
from bisect import bisect_right
import torch.optim.lr_scheduler as lr_scheduler


def make_scheduler(config: dict, optimizer: torch.optim.Optimizer, num_iters_per_epoch: int):
    """ [Creates a learning rate scheduler based on the configuration provided.]
    
        This function dynamically constructs a scheduler object for adjusting the learning rate during training. It supports custom scheduling strategies by transforming configuration parameters and adapting them  to the scheduler's expected arguments. The function handles common scheduling parameters like maximum epochs, warmup epochs, and milestone epochs, and adjusts them based on the number of iterations per epoch to accommodate different dataset sizes or training regimes.
        
        <Args>
            config (dict): 
                Configuration dictionary containing the scheduler name and its parameters. It expects 'scheduler' key with a nested 'name' key for the scheduler type, and another nested key with the same name containing specific arguments for the scheduler.
            optimizer (Optimizer): 
                The optimizer instance for which the scheduler will be applied.num_iters_per_epoch (int): The total number of iterations per epoch, used to adjust scheduling parameters.
        <Returns>
            scheduler: 
                An instance of the scheduler class specified in the configuration, initialized with the adjusted arguments. """
    # Extract scheduler name and its arguments from the config
    scheduler_name = config.get('name', None)
    scheduler_args = config.get(scheduler_name, {})
    
    # Dynamically load the scheduler class based on its name
    SchedulerClass = getattr(lr_scheduler, scheduler_name, globals().get(scheduler_name, None))
    
    # Initialize and return the scheduler instance with the filtered arguments
    scheduler = SchedulerClass(optimizer=optimizer, **scheduler_args, num_iters_per_epoch=num_iters_per_epoch, last_epoch=-1)
    
    return scheduler


class LinearWarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    """ [Implements a combined linear warmup and cosine annealing learning rate schedule.]
    
        This scheduler adjusts the learning rate using a linear warmup phase followed by a cosine annealing phase. During the warmup phase, the learning rate increases linearly from a specified initial learning rate to the optimizer's initial learning rate. After the warmup phase, it follows a cosine annealing strategy between the initial learning rate and a specified minimum learning rate over a given cycle length.
        
        <Args>
            optimizer (Optimizer):
                The optimizer for which to adjust the learning rate.
            T_max (int): 
                Maximum number of iterations after warmup, defining the cycle length for cosine annealing.
            T_warmup (int): 
                Number of iterations for the warmup phase.
            warmup_start_lr (float): 
                Starting learning rate for the warmup phase.
            eta_min (float): 
                Minimum learning rate for the cosine annealing phase.
            last_epoch (int, optional): 
                The index of the last epoch. Defaults to -1, implying the scheduler starts from the beginning.
        <Methods>
            get_lr(): 
                Computes and returns the current learning rate based on the scheduler's strategy. If called before the first epoch, it returns the starting learning rate for the warmup phase. During the warmup phase, the learning rate increases linearly. After the warmup, it follows cosine annealing.
            _get_closed_form_lr(): 
                Provides the learning rate for any epoch without requiring sequential updates. Useful for resuming training with the correct learning rate without needing to step through every epoch up to the current one. """
    
    def __init__(self, optimizer, T_max, T_warmup, warmup_start_lr, eta_min, num_iters_per_epoch, last_epoch = -1):
        self.T_max = T_max * num_iters_per_epoch
        self.T_warmup = T_warmup * num_iters_per_epoch
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning)
        # At the very start (epoch 0), initialize learning rates to the warmup start learning rate for all parameter groups.
        if self.last_epoch == 0: 
            return [self.warmup_start_lr]*len(self.base_lrs)
        # During the warmup phase, incrementally increase the learning rate linearly from warmup_start_lr to the base learning rate.
        elif self.last_epoch < self.T_warmup: 
            return [group["lr"] + (base_lr-self.warmup_start_lr)/(self.T_warmup-1) for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        # Once the warmup phase is completed, reset learning rates to their base values.
        elif self.last_epoch == self.T_warmup:
            return self.base_lrs
        # If at the start of a new cosine cycle, adjust the learning rate based on the cosine function's starting position.
        elif (self.last_epoch-1-self.T_max)%(2*(self.T_max-self.T_warmup)) == 0:
            return [group["lr"] + (base_lr-self.eta_min)*(1-math.cos(math.pi/(self.T_max-self.T_warmup)))/2 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        # Apply cosine annealing to adjust the learning rates during the annealing phase.
        else:
            return [(1+math.cos(math.pi*(self.last_epoch-self.T_warmup)/(self.T_max-self.T_warmup)))/(1+math.cos(math.pi*(self.last_epoch-self.T_warmup-1)/(self.T_max-self.T_warmup)))*(group["lr"]-self.eta_min) + self.eta_min for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        if self.last_epoch < self.T_warmup:
            return [self.warmup_start_lr + self.last_epoch*(base_lr-self.warmup_start_lr)/(self.T_warmup-1) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + 0.5*(base_lr-self.eta_min)*(1+math.cos(math.pi*(self.last_epoch-self.T_warmup)/(self.T_max-self.T_warmup))) for base_lr in self.base_lrs]


class LinearWarmupMultiStepLR(lr_scheduler._LRScheduler):
    """ [Implements a learning rate scheduler that combines a linear warmup phase with a multi-step decay.]
        
        Initially, this scheduler increases the learning rate linearly from a specified initial learning rate to the optimizer's initial learning rate during the warmup phase. After completing the warmup, it reduces the learning rate by a factor of gamma at each specified milestone.
        
        <Args>
            optimizer (Optimizer):
                The optimizer for which to adjust the learning rate.
            T_warmup (int):
                The number of iterations for the warmup phase.
            milestones (list of int):
                A list of epoch numbers at which to reduce the learning rate.
            warmup_start_lr (float):
                The starting learning rate for the warmup phase.
            gamma (float):
                The factor by which the learning rate is reduced at each milestone.
            last_epoch (int, optional):
                The index of the last epoch. Defaults to -1, implying the scheduler starts from the beginning.
        <Methods>
            get_lr():
                Computes and returns the current learning rate based on the scheduler's strategy. During the warmup phase, it linearly increases the learning rate. After the warmup, it applies a step decay based on the milestones.
                
            _get_closed_form_lr():
                Provides a way to compute the learning rate at any epoch without the need for sequential updates. This method is particularly useful for resuming training with the correct learning rate without iterating through each epoch. """
    def __init__(self, optimizer, T_warmup, milestones, warmup_start_lr, gamma, num_iters_per_epoch, last_epoch = -1):
        self.T_warmup = T_warmup * num_iters_per_epoch
        self.milestones = Counter(milestones * num_iters_per_epoch)
        self.warmup_start_lr = warmup_start_lr
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning)
        # If it's the first epoch, return the warmup start learning rate for all parameter groups.
        if self.last_epoch == 0:
            return [self.warmup_start_lr]*len(self.base_lrs)
        # If in the warmup phase, linearly increase the learning rate from warmup_start_lr to base_lr.
        elif self.last_epoch < self.T_warmup:
            return [group["lr"] + (base_lr-self.warmup_start_lr)/(self.T_warmup-1) for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        # After the warmup phase ends, reset the learning rates to their base values.
        elif self.last_epoch == self.T_warmup:
            return self.base_lrs
        # If not at a milestone, maintain the current learning rate.
        elif (self.last_epoch-self.T_warmup) not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        # Apply the decay factor at milestones.
        else:
            return [group['lr']*self.gamma**self.milestones[self.last_epoch-self.T_warmup] for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        # If still in the warmup phase, calculate the learning rate based on progress through the warmup.
        if self.last_epoch < self.T_warmup:
            return [self.warmup_start_lr + self.last_epoch*(base_lr-self.warmup_start_lr)/(self.T_warmup-1) for base_lr in self.base_lrs]
        # Calculate the effective number of step reductions to apply based on the current epoch and milestones.
        else:
            milestones = list(sorted(self.milestones.elements()))
            return [base_lr*self.gamma**bisect_right(milestones, self.last_epoch-self.T_warmup) for base_lr in self.base_lrs]