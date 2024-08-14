import os
import torch
import time
import importlib
import sys

from .model import Model

from torch.nn import functional as F
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
from libs.utils import util_system, util_engine, nms
from libs.implements import optimizers, schedulers, criterions
from libs.metrics.anetdet import ANETdetection

from torch.utils.tensorboard import SummaryWriter

@util_system.logger_wraps()
class Engine(object):
    def __init__(self, cfg: DictConfig):
        """ Default setting """
        self.cfg = cfg
        self.gpuid = tuple(map(int, self.cfg.args.gpuid.split(',')))
        self.device = torch.device(f'cuda:{self.gpuid[0]}')
        
        dataset_module = importlib.import_module(cfg.benchmark_path.replace("/", ".") + ".dataset")
        self.dataloaders = dataset_module.get_dataloaders(self.cfg, self.device) # dict, keys: 'train','valid','test'
        self.model = Model(self.cfg).to(self.device)
        self.model_ema = util_engine.ModelEma(self.model)
        
        self.optimizer = optimizers.make_optimizer(self.cfg.engine.optimizer, self.model)
        self.scheduler = schedulers.make_scheduler(self.cfg.engine.scheduler, self.optimizer, len(self.dataloaders['train']))
        
        self.evaluator = ANETdetection(
            ant_file=self.dataloaders['test'].dataset.anno_info["format"]["file_path"],
            split=self.dataloaders["test"].dataset.split[0],
            tiou_thresholds=self.dataloaders["test"].dataset.db_attributes["tiou_thresholds"])
        
        """ Load checkpoint """
        self.checkpoint_path = self.cfg.log_path + "/ckpts"
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.start_epoch = 1
        self._load_checkpoint()
        
        """ Build monitoring tools (Wandb & Tensorborad) """
        self.wandb_run = util_system.wandb_setup(self.cfg)
        self.tb_writer = SummaryWriter(os.path.join(self.cfg.log_path, "tensorboard"))
        
        # Logging files
        util_system.logging_files_to_log(self.cfg)
        
        # Calulate params & mac
        util_engine.model_params_mac_summary(
            model=self.model, 
            input=(torch.randn(1, 3200, 2304).to(self.device), torch.randn(1, 1, 2304).to(self.device)),
            metrics=['thop', 'torchinfo']
            # metrics=['ptflops']
        )
    
    def _load_checkpoint(self) -> None:
        def _load_from_file(file_path: str) -> None:
            checkpoint = torch.load(file_path, map_location=f'cuda:{self.gpuid[0]}')
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model_ema.module.load_state_dict(checkpoint['model_ema'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f"Loaded checkpoint '{file_path}' && epoch '{checkpoint['epoch']}'")
            del checkpoint
        
        if self.cfg.args.checkpoint == "latent":
            checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pt') or f.endswith('.pth') or f.endswith('.pkl')]
            if checkpoint_files:
                epochs = [int(f.split('.')[1].split('-')[0]) for f in checkpoint_files]
                latest_checkpoint_file = os.path.join(self.checkpoint_path, checkpoint_files[epochs.index(max(epochs))])
                _load_from_file(latest_checkpoint_file)
        else:
            if not os.path.isfile(self.cfg.args.checkpoint):
                logger.error(f"No checkpoint found at '{self.cfg.args.checkpoint}'")
                return
            _load_from_file(self.cfg.args.checkpoint)
    
    def _save_checkpoint_per_nth(self, nth: int, epoch: int, eval_value: float) -> None:
        if epoch % nth == 0:
            torch.save(
                {'epoch': epoch,
                 'model': self.model.state_dict(),
                 'model_ema': self.model_ema.module.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'mAP': eval_value}, 
                os.path.join(self.checkpoint_path, f"epoch.{epoch:03d}-{eval_value:.5f}.pt"))
    
    @torch.no_grad()
    def _label_points(self, points: list, gt_segments: torch.Tensor, gt_labels: torch.Tensor) -> tuple:
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self._label_points_single_video(concat_points, gt_segment, gt_label)
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    def _label_points_single_video(self, concat_points: torch.Tensor, gt_segment: torch.Tensor, gt_label: torch.Tensor) -> tuple:
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]
        gt_segment = gt_segment.to(self.device)

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.dataloaders['train'].dataset.bench_info['num_classes']), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:,1] - gt_segment[:,0]
        lens = lens[None,:].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2).to(self.device)
        left = concat_points[:,0,None] - gt_segs[:,:,0].to(self.device)
        right = gt_segs[:,:,1].to(self.device) - concat_points[:,0,None]
        reg_targets = torch.stack((left, right), dim=-1)
        
        if self.cfg.engine.center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5*(gt_segs[:, :, 0] + gt_segs[:, :, 1]).to(self.device)
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = center_pts - concat_points[:,3,None]*self.cfg.engine.center_sample_radius
            t_maxs = center_pts + concat_points[:,3,None]*self.cfg.engine.center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:,0,None] - torch.maximum(t_mins, gt_segs[:,:,0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:,:,1]) - concat_points[:,0,None]
            # F T x N x 2
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:,1,None]),
            (max_regress_distance <= concat_points[:,2,None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), 
            (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(gt_label, self.dataloaders['train'].dataset.bench_info['num_classes']).to(reg_targets.dtype)
        cls_targets = min_len_mask.to(self.device) @ gt_label_one_hot.to(self.device)
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:,3,None]

        return cls_targets, reg_targets

    def _losses(self, fpn_masks: list, out_cls_logits, out_offsets: list, gt_cls_labels: list, gt_offsets: list) -> dict:
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)
        
        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        
        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        loss_normalizer = self.cfg.engine.init_loss_norm_momentum*self.cfg.engine.init_loss_norm + (1-self.cfg.engine.init_loss_norm_momentum)*max(num_pos, 1)
        
        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]
        
        # optinal label smoothing
        gt_target *= 1-self.cfg.engine.label_smoothing
        gt_target += self.cfg.engine.label_smoothing / (self.dataloaders['train'].dataset.bench_info['num_classes']+1)
        
        # focal loss
        cls_loss = criterions.sigmoid_focal_loss(torch.cat(out_cls_logits, dim=1)[valid_mask], gt_target, reduction='sum')/loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        # giou loss defined on positive samples
        reg_loss = criterions.ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction='sum')/loss_normalizer if num_pos != 0 else 0*pred_offsets.sum()
        
        loss_weight = self.cfg.engine.loss_weight if self.cfg.engine.loss_weight > 0 else cls_loss.detach()/max(reg_loss.item(), 0.01)
        
        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        # for kl in kl_loss:
        #     final_loss=final_loss+kl
        
        losses_dict = {'cls_loss'   : cls_loss,
                       'reg_loss'   : reg_loss,
                       'final_loss' : final_loss}
        
        return losses_dict
    
    def _inference(self, batch_dict: dict, points: list, fpn_masks: list, out_cls_logits: list, out_offsets: list) -> list:
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []
        
        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(zip(batch_dict["vid_idxs"], batch_dict["vid_fps"], batch_dict["vid_lens"], batch_dict["vid_ft_stride"], batch_dict["vid_ft_nframes"])):
            # gather per-video outputs
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            # inference on a single video (should always be the case)
            results_per_vid = self._inference_single_video(points, fpn_masks_per_vid, cls_logits_per_vid, offsets_per_vid)
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)
        
        # step 3: postprocssing
        results = self._postprocessing(results)
        
        return results
    
    def _inference_single_video(self, points: list, fpn_masks: list, out_cls_logits: list, out_offsets: list) -> dict:
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []
        
        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(out_cls_logits, out_offsets, points, fpn_masks):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()
            
            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.cfg.engine.pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
            
            # 2. Keep top k top scoring boxes only
            num_topk = min(self.cfg.engine.pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()
            
            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(topk_idxs, self.dataloaders["train"].dataset.bench_info["num_classes"], rounding_mode='floor')
            cls_idxs = torch.fmod(topk_idxs, self.dataloaders["train"].dataset.bench_info["num_classes"])
            
            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]
            
            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)
            
            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.cfg.engine.duration_thresh
            
            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])
        
        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [torch.cat(x, dim=0) for x in [segs_all, scores_all, cls_idxs_all]]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}
        
        return results
    
    def _postprocessing(self, results: list) -> list:
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = float(results_per_vid['duration'])
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            
            if self.cfg.engine.nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = nms.batched_nms(
                    segs, scores, labels,
                    self.cfg.engine.iou_threshold,
                    self.cfg.engine.min_score,
                    self.cfg.engine.max_seg_num,
                    use_soft_nms = (self.cfg.engine.nms_method  == 'soft'),
                    multiclass = self.cfg.engine.multiclass_nms,
                    sigma = self.cfg.engine.nms_sigma,
                    voting_thresh = self.cfg.engine.voting_thresh
                )
            
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs*stride + 0.5*nframes)/fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen]*0.0 + vlen
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )
        
        return processed_results
    
    @util_system.logger_wraps()
    def _train(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> dict:
        self.model.train()
        batch_time = util_engine.AverageMeter()
        losses_tracker = {}
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True, initial=1, miniters=self.cfg.engine.print_freq)
        start = time.time()
        for batch_dict in dataloader:
            """ [Train dataloader's batch_dict format]
                batch_dict = {
                    'batched_inputs' : batched_inputs,
                    'batched_masks'  : batched_masks,
                    'gt_segments'    : gt_segments,
                    'gt_labels'      : gt_labels} """
            # zero out optim
            self.optimizer.zero_grad(set_to_none=True)
            # forward / backward the model
            points, fpn_masks, out_cls_logits, out_offsets = self.model(batch_dict['batched_inputs'].to(self.device), batch_dict['batched_masks'].to(self.device))
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self._label_points(points, batch_dict['gt_segments'], batch_dict['gt_labels'])
            
            # compute the loss and return
            losses = self._losses(fpn_masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets)
            losses['final_loss'].backward()
            
            # gradient cliping (to stabilize training if necessary)
            if self.cfg.engine.clip_grad_l2norm > 0.0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.engine.clip_grad_l2norm)
            
            # step optimizer / scheduler
            self.optimizer.step()
            self.scheduler.step()
            self.model_ema.update()
            
            # printing (only check the stats when necessary to avoid extra cost)
            if pbar.n % self.cfg.engine.print_freq == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time()-start)/self.cfg.engine.print_freq)
                start = time.time()
                # track all losses
                for key, value in losses.items():
                    # init meter if necessary
                    if key not in losses_tracker: losses_tracker[key] = util_engine.AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())
                # log to tensor board
                lr = self.scheduler.get_last_lr()[0]
                global_step = epoch*len(dataloader) + pbar.n
                # learning rate (after stepping)
                self.tb_writer.add_scalar('train/learning_rate', lr, global_step)
                # all losses
                tag_dict = {key: value.val for key, value in losses_tracker.items() if key != "final_loss"}
                self.tb_writer.add_scalars('train/all_losses', tag_dict, global_step)
                # final loss
                self.tb_writer.add_scalar('train/final_loss', losses_tracker['final_loss'].val, global_step)
                # print to terminal
                summary_per_freq = {
                    'Epoch': f'{epoch}',
                    'Time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                    'Loss': f'{losses_tracker["final_loss"].val:.2f}({losses_tracker["final_loss"].avg:.2f})'
                }
                summary_per_freq.update({key: f'{value.val:.2f} ({value.avg:.2f}' for key, value in losses_tracker.items()})
                pbar.set_postfix(summary_per_freq)
                
            pbar.update(1)
        pbar.close()
        return losses
    
    
    @util_system.logger_wraps()
    @torch.inference_mode()
    def _validate(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> dict:
        self.model.eval()
        batch_time = util_engine.AverageMeter()
        losses_tracker = {}
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="RED", dynamic_ncols=True, initial=1, miniters=self.cfg.engine.print_freq)
        start = time.time()
        for batch_dict in dataloader:
            """ [Valid dataloader's batch_dict format]
                batch_dict = {
                    'batched_inputs' : batched_inputs,
                    'batched_masks'  : batched_masks,
                    'gt_segments'    : gt_segments,
                    'gt_labels'      : gt_labels} """
            # zero out optim
            self.optimizer.zero_grad(set_to_none=True)
            # forward / backward the model
            points, fpn_masks, out_cls_logits, out_offsets = self.model(batch_dict["batched_inputs"].to(self.device), batch_dict["batched_masks"].to(self.device))
            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self._label_points(points, batch_dict["gt_segments"].to(self.device), batch_dict["gt_labels"].to(self.device))
            
            # compute the loss and return
            losses = self._losses(fpn_masks, out_cls_logits, out_offsets,gt_cls_labels, gt_offsets)

            # printing (only check the stats when necessary to avoid extra cost)
            if pbar.n % self.cfg.engine.print_freq == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time()-start)/self.cfg.engine.print_freq)
                start = time.time()
                # track all losses
                for key, value in losses.items():
                    # init meter if necessary
                    if key not in losses_tracker: losses_tracker[key] = util_engine.AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())
                # log to tensor board
                lr = self.scheduler.get_last_lr()[0]
                global_step = epoch*len(dataloader) + pbar.n
                # learning rate (after stepping)
                self.tb_writer.add_scalar('train/learning_rate', lr, global_step)
                # all losses
                tag_dict = {key: value.val for key, value in losses_tracker.items() if key != "final_loss"}
                self.tb_writer.add_scalars('train/all_losses', tag_dict, global_step)
                # final loss
                self.tb_writer.add_scalar('train/final_loss', losses_tracker['final_loss'].val, global_step)
                # print to terminal
                summary_per_freq = {
                    'Epoch': f'{epoch}',
                    'Time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                    'Loss': f'{losses_tracker["final_loss"].val:.2f}({losses_tracker["final_loss"].avg:.2f})'
                }
                summary_per_freq.update({key: f'{value.val:.2f} ({value.avg:.2f}' for key, value in losses_tracker.items()})
                pbar.set_postfix(summary_per_freq)
            pbar.update(1)
        pbar.close()
        return losses

        
    @util_system.logger_wraps()
    @torch.inference_mode()
    def _test(self, dataloader: torch.utils.data.DataLoader, evaluator: ANETdetection) -> float:
        # self.model.eval()
        self.model_ema.module.eval()
        # set up meters
        batch_time = util_engine.AverageMeter()
        results = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="GREEN", dynamic_ncols=True, initial=1, miniters=self.cfg.engine.print_freq)
        start = time.time()
        for batch_dict in dataloader:
            """ [Test dataloader's batch_dict format]
                batch_dict = {
                    'batched_inputs' : batched_inputs,
                    'batched_masks'  : batched_masks,
                    'vid_idxs'       : vid_idxs,
                    'vid_fps'        : vid_fps,
                    'vid_lens'       : vid_lens,
                    'vid_ft_stride'  : vid_ft_stride, 
                    'vid_ft_nframes' : vid_ft_nframes} """
            # points, fpn_masks, out_cls_logits, out_offsets = self.model(batch_dict["batched_inputs"].to(self.device), batch_dict["batched_masks"].to(self.device))
            points, fpn_masks, out_cls_logits, out_offsets = self.model_ema.module(batch_dict["batched_inputs"].to(self.device), batch_dict["batched_masks"].to(self.device))
            output = self._inference(batch_dict, points, fpn_masks, out_cls_logits, out_offsets)
            
            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend([output[vid_idx]['video_id']]*output[vid_idx]['segments'].shape[0])
                    results['t-start'].append(output[vid_idx]['segments'][:,0])
                    results['t-end'].append(output[vid_idx]['segments'][:,1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])
            # printing
            if pbar.n % (self.cfg.engine.print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time()-start)/self.cfg.engine.print_freq)
                start = time.time()
                # print to terminal
                summary_per_freq = {'Time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'}
                pbar.set_postfix(summary_per_freq)
            pbar.update(1)
        pbar.close()
        # gather all stats and evaluate
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()

        if evaluator is not None:
            if (self.cfg.engine.ext_score_file is not None) and isinstance(self.cfg.engine.ext_score_file, str):
                results = util_engine.postprocess_results(results, self.cfg.engine.ext_score_file)
        
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)

        # log mAP to tb_writer
        self.tb_writer.add_scalar('validation/mAP', mAP, -1)

        return mAP
    
    @util_system.logger_wraps()
    def run(self) -> None:
        with torch.cuda.device(self.device):
            if self.wandb_run: self.wandb_run.watch(self.model, log="all")
            if self.cfg.args.mode == "train":
                init_valid_loss = 0
                for epoch in range(self.start_epoch, self.cfg.engine.max_epochs):
                    valid_loss_best = init_valid_loss
                    train_start_time = time.time()
                    train_losses = self._train(self.dataloaders['train'], epoch)
                    train_end_time = time.time()
                    train_speed = (train_end_time-train_start_time)/(len(self.dataloaders['train'])*self.dataloaders['train'].batch_size)
                    logger.info(f"\n\t[GPU {self.gpuid}] Wandb run name: {self.wandb_run.name}\n\t[TRAIN] - Epoch {epoch}: Loss_final={train_losses['final_loss']:.4f}, Loss_cls={train_losses['cls_loss']:.4f}, Loss_reg={train_losses['reg_loss']:.4f}, Speed={train_speed:.4f}(s/video)")
                    
                    if self.dataloaders.get('valid', None):
                        valid_start_time = time.time()
                        valid_losses = self._validate(self.dataloaders['valid'], epoch)
                        valid_end_time = time.time()
                        valid_speed = (valid_end_time-valid_start_time)/(len(self.dataloaders['valid'])*self.dataloaders['valid'].batch_size)
                        logger.info(f"\n\t[GPU {self.gpuid}] Wandb run name: {self.wandb_run.name}\n\t[VALID] - Epoch {epoch}: Loss_final={valid_losses['final_loss']:.4f}, Loss_cls={valid_losses['cls_loss']:.4f}, Loss_reg={valid_losses['reg_loss']:.4f}, Speed={valid_speed:.4f}(s/video)")
                    
                    # if epoch >= 5 and epoch % 2 == 0:
                    if epoch >= 25:
                        test_start_time = time.time()
                        mAP = self._test(self.dataloaders['test'], self.evaluator)
                        test_end_time = time.time()
                        test_speed = (test_end_time-test_start_time)/(len(self.dataloaders['test'])*self.dataloaders['test'].batch_size)
                        logger.info(f"\n\t[GPU {self.gpuid}] Wandb run name: {self.wandb_run.name}\n\t[TEST] - Epoch {epoch}: mAP={mAP:.4f}, Speed={test_speed:.4f}(s/video)")
                        
                    test_mAP = locals().get('mAP', 0)
                    test_speed = locals().get('test_speed', 0)
                    results = {
                        'Learning Rate': self.optimizer.param_groups[0]['lr'],
                        'Train Final Loss': train_losses['final_loss'], 
                        'Train Cls Loss': train_losses['cls_loss'],
                        'Train Reg Loss': train_losses['reg_loss'],
                        'Train Speed': train_speed,
                        # 'Valid Final Loss': valid_losses['final_loss'], 
                        # 'Valid Cls Loss': valid_losses['cls_loss'], 
                        # 'Valid Reg Loss': valid_losses['reg_loss'], 
                        # 'Valid Speed': valid_speed,
                        'Test mAP': test_mAP,
                        'Test Speed': test_speed}
                    
                    if epoch >= 25:
                        self._save_checkpoint_per_nth(nth=1, epoch=epoch, eval_value=test_mAP)
                    
                    if self.wandb_run: self.wandb_run.log(results)
                self.tb_writer.close()
                logger.info(f"Training for {self.cfg.engine.max_epochs} epoches done!")
            else:
                test_start_time = time.time()
                mAP = self._test(self.dataloaders['test'], self.evaluator)
                test_end_time = time.time()
                test_speed = (test_end_time-test_start_time)/(len(self.dataloaders['test'])*self.dataloaders['test'].batch_size)
                
                logger.info(f"\n [TEST] : mAP={mAP:.4f} | Speed={test_speed:.2f}(s/video)")