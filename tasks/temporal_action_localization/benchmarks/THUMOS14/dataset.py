import os
import io
import json
import torch
import mmengine
import numpy as np

from omegaconf import DictConfig
from functools import partial
from torch.nn import functional as F
from typing import Tuple, Dict
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from libs.utils import util_system, util_dataset


@util_system.logger_wraps()
def get_dataloaders(cfg: DictConfig, device: torch.device) -> Dict:    
    dataloaders = {}
    partitions = ["train", "test"] if cfg.args.mode == "train" else ["test"]
    
    for partition in partitions:
        dataset = THUMOS14(
            partition = partition, 
            bench_info = cfg.dataset.bench_info, # Benchmark information (dict)
            anno_info = cfg.dataset.anno_info, # Annotation information (dict)
            feat_info = cfg.dataset.feat_info # Feature information (dict)
        )
        if partition == "train": 
            for head_name in cfg.model.head_info.name:
                head_args = cfg.model.head_info.get(head_name, {})
                if head_args.get('head_empty_cls', None): head_args.head_empty_cls = dataset.get_attributes()['empty_label_ids']
        dataloader = DataLoader(
            dataset = dataset,
            pin_memory = cfg.dataset.loader.pin_memory, 
            num_workers = cfg.dataset.loader.num_workers,
            worker_init_fn = util_dataset.worker_init_reset_seed if partition == 'train' else None,
            persistent_workers = True,
            batch_size = cfg.dataset.loader.batch_size if partition == 'train' else 1,
            shuffle = (partition == 'train'),
            drop_last = (partition == 'train'),
            generator = util_dataset.fix_random_seed(seed=cfg.dataset.loader.seed, include_cuda = True),
            collate_fn = partial(_collate_fn, partition=partition, args=cfg.dataset.loader, device=device)
        )
        dataloaders[partition] = dataloader
    return dataloaders


def _collate_fn(batch, partition: str, args: DictConfig, device: torch.device):
    """ [Dataloader's __getitem__() result format]
        data_dict = {'video_id'       : self.data_list[idx]['id'],
                    'fps'             : self.data_list[idx]['fps'],
                    'duration'        : self.data_list[idx]['duration'],
                    'feat_num_frames' : self.feat_info['meta']['num_frames'],
                    'labels'          : labels,     # N
                    'segments'        : segments,   # N x 2
                    'feats'           : feats,      # C x T
                    'feat_stride'     : feat_stride} """
    
    # Input data preprocess
    feats = [item['feats'] for item in batch] # Extract features from the batch
    feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats]) # Calculate the lengths of all feature sequences
    max_len = feats_lens.max(0).values.item() # Find the maximum sequence length in this batch
    
    max_seq_len = args.max_seq_len
    padding_value = args.padding_value
    max_div_factor = args.max_div_factor
    
    if partition != "test":
        assert max_len <= max_seq_len, "Input length must be smaller than max_seq_len during training"
        max_len = max_seq_len # set max_len to max_seq_len
        batch_shape = (len(feats), feats[0].shape[0], max_len) # Batch input shape: B, C, T
        batched_inputs = feats[0].new_full(batch_shape, padding_value)
        for feat, pad_feat in zip(feats, batched_inputs): 
            pad_feat[..., :feat.shape[-1]].copy_(feat) # zero padding to feat, T=max_len;2304
        
        # Generate the mask - True: valid, False: padding
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        batched_masks = batched_masks.unsqueeze(1)
        
        # Label preprocess
        # generate segment/lable List[N x 2] / List[N] with length = B
        assert batch[0]['segments'] is not None, "GT action labels does not exist"
        assert batch[0]['labels'] is not None, "GT action labels does not exist"
        # print(video_list)
        gt_segments = [item['segments'] for item in batch]
        gt_labels = [item['labels'] for item in batch]
        
        batch_dict = {'batched_inputs' : batched_inputs,
                      'batched_masks'  : batched_masks,
                      'gt_segments'    : gt_segments,
                      'gt_labels'      : gt_labels}
        return batch_dict
    else:
        assert len(batch) == 1, "Only support batch_size = 1 during inference"
        max_len = max_seq_len if max_len <= max_seq_len else (max_len+(max_div_factor-1))//max_div_factor * max_div_factor # Pad the input to the next divisible size
        padding_size = [0, max_len - feats_lens[0]]
        batched_inputs = F.pad(feats[0], padding_size, value=padding_value).unsqueeze(0)
        # Generate the mask
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_masks = torch.arange(max_len)[None,:] < feats_lens[:,None]
        batched_masks = batched_masks.unsqueeze(1)
        
        # Label preprocess
        vid_idxs = [item['video_id'] for item in batch]
        vid_fps = [item['fps'] for item in batch]
        vid_lens = [item['duration'] for item in batch]
        vid_ft_stride = [item['feat_stride'] for item in batch]
        vid_ft_nframes = [item['feat_num_frames'] for item in batch]
        
        batch_dict = {'batched_inputs' : batched_inputs,
                      'batched_masks'  : batched_masks,
                      'vid_idxs'       : vid_idxs,
                      'vid_fps'        : vid_fps,
                      'vid_lens'       : vid_lens,
                      'vid_ft_stride'  : vid_ft_stride,
                      'vid_ft_nframes' : vid_ft_nframes}
        return batch_dict


def remove_unuseful_annotations(ants):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        if s < e: valid_events.append(event)
    return valid_events


class THUMOS14(Dataset):
    def __init__(self, partition: str, bench_info: dict, anno_info: dict, feat_info: dict):
        self.partition = partition
        self.bench_info = bench_info
        self.anno_info = anno_info
        self.feat_info = feat_info
        self.split = ['validation'] if self.partition == 'train' else ['test']
        
        # load database and select the subset
        self.data_list, self.label_dict = self._load_json_db(self.anno_info['format']['file_path'])
        
        assert len(self.label_dict) == self.bench_info['num_classes']
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [], # we will mask out cliff diving
        }
    
    def get_attributes(self):
        return self.db_attributes
    
    def _load_json_db(self, path: str) -> Tuple[tuple, Dict]:
        # load database and select the subset
        with open(path, 'r') as fid: json_data = json.load(fid)
        json_db = json_data['database']
        
        # if label_dict is not available
        label_dict = {}
        for key, value in json_db.items():
            for act in value['annotations']:
                label_dict[act['label']] = act['label_id']
        
        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split: continue
            
            # get fps if available
            if self.feat_info['meta']['default_fps']: fps = self.feat_info['meta']['default_fps']
            elif 'fps' in value: fps = value['fps']
            else: assert False, "Unknown video FPS."
            
            # get video duration if available
            duration = value['duration'] if 'duration' in value else 1e8
            
            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7). our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])
                segments = torch.tensor(segments, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64).squeeze(1)
            else:
                segments, labels = None, None
            
            dict_db += ({'id': key, 
                         'fps': fps, 
                         'duration': duration, 
                         'segments': segments,
                         'labels': labels},)
        return dict_db, label_dict
    
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, idx: int):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data

        #TODO: Integrate numpy process to tensor calculation
        feat_fp = os.path.join(
            self.feat_info['format']['dir_path'], 
            self.feat_info['format']['prefix'] + 
            self.data_list[idx]['id'] + 
            self.feat_info['format']['type'] + 
            self.feat_info['format']['ext']
        )
        if "npy" in self.feat_info['format']['ext']:
            data = io.BytesIO(mmengine.get(feat_fp))
            feats = np.load(data).astype(np.float32)
        elif "pt" in self.feat_info['format']['ext']:
            data = io.BytesIO(mmengine.get(feat_fp))
            feats = torch.load(data) # T x C, C=3200;Backbone_in_dim
        else:
            raise NotImplementedError
        feats = feats[::self.feat_info['meta']['downsample_rate'],:] # 
        feats = feats.permute(1, 0).contiguous() # T x C -> C x T
        feat_stride = self.feat_info['meta']['feat_stride'] * self.feat_info['meta']['downsample_rate']
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if self.data_list[idx]['segments'].any():
            segments = (self.data_list[idx]['segments']*self.data_list[idx]['fps'] - 0.5*self.feat_info['meta']['num_frames']) / feat_stride
            labels = self.data_list[idx]['labels']
        else: segments, labels = None, None
        
        # return a data dict
        data_dict = {'video_id'        : self.data_list[idx]['id'],
                     'fps'             : self.data_list[idx]['fps'],
                     'duration'        : self.data_list[idx]['duration'],
                     'feat_num_frames' : self.feat_info['meta']['num_frames'],
                     'labels'          : labels,     # N
                     'segments'        : segments,   # N x 2
                     'feats'           : feats,      # C x T
                     'feat_stride'     : feat_stride}
        
        # truncate the features during training
        if self.partition == 'train' and (segments is not None): data_dict = util_dataset.truncate_feats(data_dict, self.feat_info['meta']['max_seq_len'], self.feat_info['meta']['trunc_thresh'], self.feat_info['meta']['crop_ratio'])
        
        return data_dict