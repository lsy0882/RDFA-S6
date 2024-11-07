# RDFA-S6

## Guide
### [Instructions for Environment Setting]

#### Step 1: Dependency setting
⭐NOTE
* We suggest using [`conda`](https://docs.conda.io/en/latest/) to manage your packages.
* To prevent conflicts between dependencies, the environment was not exported to a `.yaml` or `.txt` file. Therefore, you will need to manually install the required packages according to the guidelines.
* Using gpu version of pytorch will significantly accelerate the feature extraction procedure. Please refer to [here](https://pytorch.org/get-started/locally/) for more detailed settings.
```bash
# Clone git
git clone https://github.com/lsy0882/RDFA-S6.git

# Create conda environments
conda create -n rdfa-s6 python=3.11
conda activate rdfa-s6

# Install pytorch 
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Build cmake libraries
cd ./build/causal-conv1d/
python setup.py develop
cd ../mamba/
python setup.py develop
cd ../nms_1d_cpu/
python setup.py install --user
cd ../..

# Install packages
pip install hydra-core
pip install wandb
pip install einops
pip install torchinfo ptflops thop
pip install pandas joblib
pip install tensorboard
pip install mmengine
```

#### Step 2: Configuration setting
⭐NOTE
* This project uses `Hydra` to manage configuration files (`.yaml`). The configuration files are structured into four types, with their respective locations and roles as follows:<br>
1. run.yaml
   * Location: `./`
   * Role: Configuration file for global variables, Hydra, Wandb, and system settings.
   * <details>
     <summary>Sample</summary>

     ```yaml
     args:
         task: temporal_action_localization
         benchmark: THUMOS14 # Select one from the following options: [ActivityNet, FineAction, HACS, THUMOS14]
         model: RDFA-S6 # Choose the directory name located in tasks/${task}/models/
         mode: train # Select either "train" or "test"
         exp_name: b2_me50_ResidualSharedBiMambaBackbone_v1.19.0-10 # Enter the name of your experiment
         checkpoint: latent # Choose either "latent" or specify your weight path
         gpuid: "0" # Specify your GPU index (only a single GPU is supported)
  
     model_path: tasks/${args.task}/models/${args.model} #! Don't change
     log_path: ${model_path}/logs/${args.benchmark}/${args.exp_name}/ #! Don't change
     benchmark_path: tasks/${args.task}/benchmarks/${args.benchmark} #! Don't change
  
     engine: ${model_path}/engine #! Don't change
     dataset: ${benchmark_path}/dataset #! Don't change
     model: ${model_path}/model #! Don't change
  
     wandb:
         login:
             key: "" #! Insert your wandb personal API key
         init: # Reference: https://docs.wandb.ai/ref/python/init
             entity: "" #! Insert your wandb profile name or team name
             project: "[Project] RSMamba-TAL-Dev"
             name: ${args.benchmark}-${args.model}-${args.exp_name}
             id: ${args.benchmark}-${args.model}-${args.exp_name}
             job_type: ${args.mode}
             group: 
             tags: ["${args.benchmark}", "${args.model}"]
             notes: "RS-Mamba initial update"
             dir: ${log_path}/ #! Don't change
             resume: "auto"
             save_code: true
             reinit: false
             magic: ~
             config_exclude_keys: []
             config_include_keys: []
             anonymous:
             mode: "online"
             allow_val_change: true
             force: false
             sync_tensorboard: false
             monitor_gym: false
  
     hydra:
         run:
             dir: ${log_path}/outputs #! Don't change
         job_logging:
             level: INFO #! Don't change
         sweep:
             dir: ${log_path}/multirun #! Don't change
  
     job:
         name: ${args.exp_name} #! Don't change
         id: ${args.exp_name} #! Don't change
         num:
         config_path:
         config_name:
  
     loguru:
         handlers:
             - sink: ${log_path}/loguru.log #! Don't change
               level: DEBUG #! Don't change
               format: "{time} {level} {message}" #! Don't change
    </details>
2. dataset.yaml
   * Location: `./tasks/${args.task}/benchmarks/${args.benchmark}`
   * Role: Configuration file for data preprocessing and batching-related settings.
   * <details>
     <summary>Sample</summary>

     ```yaml
     dataset:
         bench_info:
             num_classes: 20 # Adjust the value according to the number of classes handled by the benchmark.
         anno_info:
             format:
                 file_path: "" # Insert the file path for the annotation.
         feat_info:
             format:
                 dir_path: "" # Insert the directory path where the features are located.
                 prefix: "" # Define this variable if you are using a prefix during preprocessing.
                 type: "" # Define this variable if you are using a mid-term value during preprocessing.
                 ext: "" # Define this variable if you are using an extension during preprocessing.
             meta: # Define and utilize preprocessing variables for the data.
                 feat_stride: 4
                 downsample_rate: 1
                 num_frames: 16
                 default_fps: ~
                 max_seq_len: 2304
                 trunc_thresh: 0.5
                 crop_ratio: [0.9, 1.0]
         loader: # Set up the configurations related to the dataloader.
             pin_memory: false
             num_workers: 20
             seed: 1234567891
             batch_size: 2
             max_seq_len: ${dataset.feat_info.meta.max_seq_len}
             padding_value: 0.0
             max_div_factor: 1
    </details>
3. model.yaml
   * Location: `./tasks/${args.task}/models/${args.model}`
   * Role: Configuration file for architecture modeling-related settings.
   * <details>
     <summary>Sample</summary>

     ```yaml
     model:
         backbone_info:
             name: ResidualSharedBiMambaBackbone
             ResidualSharedBiMambaBackbone:
                 EmbeddingModule:
                     input_c: 3200
                     emb_c: 512
                     kernel_size: 3
                     stride: 1
                     padding: ${floordiv:${model.backbone_info.ResidualSharedBiMambaBackbone.EmbeddingModule.kernel_size}, 2}
                     dilation: 1
                     groups: 1
                     bias: false
                     padding_mode: "zeros"
                StemModule:
                    block_n: 1
                    emb_c: ${model.backbone_info.ResidualSharedBiMambaBackbone.EmbeddingModule.emb_c}
                    kernel_size: 4
                    drop_path_rate: 0.3
                    recurrent: 4
                BranchModule:
                    block_n: 5
                    emb_c: ${model.backbone_info.ResidualSharedBiMambaBackbone.EmbeddingModule.emb_c}
                    kernel_size: 4
                    drop_path_rate: 0.3
         neck_info:
             name: FPNIdentity
             FPNIdentity:
                 in_channels: 512
                 out_channel: 512
                 with_ln: true
                 scale_factor: 2
             FPN1D:
                 in_channels: 512
                 out_channel: 512
                 with_ln: true
                 scale_factor: 2
         generator_info:
             name: PointGenerator
             PointGenerator:
                 max_seq_len: 2304
                 max_buffer_len_factor: 6.0
                 scale_factor: 2
                 fpn_levels: # TBD
                 regression_range: [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]]
         head_info:
             name:
                 - PtTransformerClsHead
                 - PtTransformerRegHead
             PtTransformerClsHead:
                 input_dim: 512 # fpn_dim
                 feat_dim: 512 # head_dim
                 num_classes: 20
                 prior_prob: 0.01
                 num_layers: 3
                 kernel_size: 3
                 with_ln: true
                 empty_cls: []
             PtTransformerRegHead:
                 input_dim: 512 # fpn_dim
                 feat_dim: 512 # head_dim
                 fpn_levels: # TBD
                 num_layers: 3
                 kernel_size: 3
                 with_ln: true
    </details>
4. engine.yaml
   * Location: `./tasks/${args.task}/models/${args.model}`
   * Role: Configuration file for train/infer-related settings for the target model.
   * <details>
     <summary>Sample</summary>

     ```yaml
     engine:
         max_epochs: 50
         clip_grad_l2norm: 1.0
         print_freq: 5
         center_sample: radius
         center_sample_radius: 1.5
         init_loss_norm: 100
         init_loss_norm_momentum: 0.9
         label_smoothing: 0.0
         loss_weight: 1.0
         pre_nms_thresh: 0.001
         pre_nms_topk: 2000
         duration_thresh: 0.05
         nms_method: soft
         iou_threshold: 0.1
         min_score: 0.001
         max_seg_num: 200
         multiclass_nms: true
         nms_sigma: 0.5
         voting_thresh: 0.7
         ext_score_file:
         criterion:
             name: loss1
         optimizer: 
             name: AdamW
             SGD:
                 lr: 1.0e-4
                 momentum: 0.9
                 weight_decay: 5.0e-2
             AdamW:
                 lr: 1.0e-4
                 weight_decay: 5.0e-2
         scheduler: 
             name: LinearWarmupCosineAnnealingLR
             LinearWarmupCosineAnnealingLR:
                 T_max: ${engine.max_epochs}
                 T_warmup: 5
                 warmup_start_lr: 0.0
                 eta_min: 1e-8
             LinearWarmupMultiStepLR:
                 T_warmup: 5
                 milestones: [30, 60, 90]
                 warmup_start_lr: 0.0
                 gamma: 0.1
             CosineAnnealingLR:
                 max_epochs: ${engine.max_epochs}
                 eta_min: 0
             MultiStepLR:
                 milestone_epochs: []
                 gamma: 0.1
    </details>

### [Instructions for Running the Engine]
⭐NOTE
* This project provides monitoring with `TensorBoard` by default, and optionally with [`wandb`](https://www.wandb.com/). If the API key for wandb is not provided in the settings, `wandb` will not be used, and only `TensorBoard` will be available for monitoring.<br>

#### Step 1: Set up configuration files
Set up the configuration by referring to the comments in each configuration file.

#### Step 2: Run target model's engine
Once the configuration setup is complete, run the `run.py` as follows.
```bash
python run.py
```

### [Pretrained Weights Download]
Link: [`Dropbox`](https://www.dropbox.com/scl/fi/r8ai5ekkws5ezoghwa11c/thumos_best.pt?rlkey=5is4to9v4czgbae9va3pl8mbx&st=363ym4zb&dl=0)

## Citation
If you use this code or dataset in your research, please cite our paper:
```bibtex
@article{lee2024enhancing,
  title={Enhancing Temporal Action Localization: Advanced S6 Modeling with Recurrent Mechanism},
  author={Lee, Sangyoun and Jung, Juho and Oh, Changdae and Yun, Sunghee},
  journal={arXiv preprint arXiv:2407.13078},
  year={2024}
}
