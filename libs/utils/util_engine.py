import os
import torch
import numpy as np
import json
from loguru import logger
from copy import deepcopy

from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile


class ModelEma(torch.nn.Module):
    def __init__(self, original_model, decay=0.999):
        super().__init__()
        self.original_model = original_model
        self.module = deepcopy(self.original_model).eval() # make a copy of the model for accumulating moving average of weights
        self.decay = decay
    
    def update(self):
        with torch.no_grad():
            ema_state_dict = self.module.state_dict()
            for name, param in self.original_model.named_parameters():
                ema_param = ema_state_dict[name]
                ema_param.data.copy_(self.decay * ema_param.data + (1.0 - self.decay) * param.data)


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_results(results, cls_score_file, num_pred=200, topk=2):
    
    def results_to_array(results, num_pred):
        # video ids and allocate the dict
        vidxs = sorted(list(set(results['video-id'])))
        results_dict = {}
        for vidx in vidxs:
            results_dict[vidx] = {
                'label'   : [],
                'score'   : [],
                'segment' : [],
            }

        # fill in the dict
        for vidx, start, end, label, score in zip(
            results['video-id'],
            results['t-start'],
            results['t-end'],
            results['label'],
            results['score']
        ):
            results_dict[vidx]['label'].append(int(label))
            results_dict[vidx]['score'].append(float(score))
            results_dict[vidx]['segment'].append(
                [float(start), float(end)]
            )

        for vidx in vidxs:
            label = np.asarray(results_dict[vidx]['label'])
            score = np.asarray(results_dict[vidx]['score'])
            segment = np.asarray(results_dict[vidx]['segment'])

            # the score should be already sorted, just for safety
            inds = np.argsort(score)[::-1][:num_pred]
            label, score, segment = label[inds], score[inds], segment[inds]
            results_dict[vidx]['label'] = label
            results_dict[vidx]['score'] = score
            results_dict[vidx]['segment'] = segment

        return results_dict
    
    def load_results_from_json(filename):
        assert os.path.isfile(filename)
        with open(filename, "r") as f:
            results = json.load(f)
        # for activity net external classification scores
        if 'results' in results:
            results = results['results']
        return results

    # load results and convert to dict
    # if isinstance(results, str):
    #     results = load_results_from_pkl(results)
    # array -> dict
    results = results_to_array(results, num_pred)

    # load external classification scores
    if '.json' in cls_score_file:
        cls_scores = load_results_from_json(cls_score_file)
    # else:
    #     cls_scores = load_results_from_pkl(cls_score_file)

    # dict for processed results
    processed_results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # process each video
    for vid, result in results.items():

        # pick top k cls scores and idx
        if len(cls_scores[vid])==1:
            curr_cls_scores = np.asarray(cls_scores[vid][0])
        else:
            curr_cls_scores = np.asarray(cls_scores[vid])

        if max(curr_cls_scores)>1 or min(curr_cls_scores)<0:
            curr_cls_scores=softmax(curr_cls_scores)
        
        topk_cls_idx = np.argsort(curr_cls_scores)[::-1][:topk]
        topk_cls_score = curr_cls_scores[topk_cls_idx]

        # model outputs
        pred_score, pred_segment, pred_label = \
            result['score'], result['segment'], result['label']
        num_segs = min(num_pred, len(pred_score))

        # duplicate all segment and assign the topk labels
        # K x 1 @ 1 N -> K x N -> KN
        # multiply the scores
        # temp = np.abs(topk_cls_score[:, None] @ pred_score[None, :])
        # new_pred_score = np.sqrt(temp).flatten()        
        new_pred_score = np.sqrt(topk_cls_score[:, None] @ pred_score[None, :]).flatten()
        new_pred_segment = np.tile(pred_segment, (topk, 1))
        new_pred_label = np.tile(topk_cls_idx[:, None], (1, num_segs)).flatten()

        # add to result
        processed_results['video-id'].extend([vid]*num_segs*topk)
        processed_results['t-start'].append(new_pred_segment[:, 0])
        processed_results['t-end'].append(new_pred_segment[:, 1])
        processed_results['label'].append(new_pred_label)
        processed_results['score'].append(new_pred_score)
        # pdb.set_trace()

    processed_results['t-start'] = np.concatenate(
        processed_results['t-start'], axis=0)
    processed_results['t-end'] = np.concatenate(
        processed_results['t-end'], axis=0)
    processed_results['label'] = np.concatenate(
        processed_results['label'],axis=0)
    processed_results['score'] = np.concatenate(
        processed_results['score'], axis=0)

    return processed_results


def save_checkpoint_per_nth(nth, epoch, model, optimizer, train_loss, valid_loss, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if epoch % nth == 0:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # Log and save the checkpoint file using wandb
        wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))

def save_checkpoint_per_best(best, valid_loss, train_loss, epoch, model, optimizer, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if valid_loss < best:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # # Log and save the checkpoint file using wandb
        # wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        best = valid_loss
    return best

def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    # Add another schedulers
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        logger.info(f"{name}: {param_count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model, input, metrics):
    
    # ptflops
    if 'ptflops' in metrics:
        input_shape = ((input[0].shape[-2], input[0].shape[-1]), (input[1].shape[-2], input[1].shape[-1]))
        MACs_ptflops, params_ptflops = get_model_complexity_info(model, input_shape, print_per_layer_stat=False, verbose=False) 
        MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
        logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")

    # thop
    if 'thop' in metrics:
        MACs_thop, params_thop = profile(model, inputs=(input[0], input[1]), verbose=False)
        MACs_thop, params_thop = MACs_thop / 1e6, params_thop / 1e6
        logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
    
    # torchinfo
    if 'torchinfo' in metrics:
        model_profile = summary_(model, input_size=[input[0].shape, input[1].shape])
        MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds / 1e6, model_profile.total_params / 1e6
        logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")