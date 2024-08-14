# RDFA-S6

## Guide
### 1. Instructions for Environment Setting

#### Step 1: Dependency setting
⭐ We suggest using [`conda`](https://docs.conda.io/en/latest/) to manage your packages.
⭐ To prevent conflicts between dependencies, the environment was not exported to a `.yaml` or `.txt` file. Therefore, you will need to manually install the required packages according to the guidelines.
⭐ Using gpu version of pytorch will significantly accelerate the feature extraction procedure. Please refer to [here](https://pytorch.org/get-started/locally/) for more detailed settings.
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

Click to expand</summary>


#### Step 2: Configuration setting
⭐ This project uses `Hydra` to manage configuration files (`.yaml`). The configuration files are structured into four types, with their respective locations and roles as follows:
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
         ```
</details>
2. dataset.yaml
   * Location: `./tasks/${args.task}/benchmarks/${args.benchmark}`
   * Role: Configuration file for data preprocessing and batching-related settings.
   * Sample:
     ```yaml
     
     ```
3. model.yaml
   * Location: `./tasks/${args.task}/models/${args.model}`
   * Role: Configuration file for architecture modeling-related settings.
   * Sample:
     ```yaml
     
     ```
4. engine.yaml
   * Location: `./tasks/${args.task}/models/${args.model}`
   * Role: Configuration file for train/infer-related settings for the target model.
   * Sample:
     ```yaml
     
     ```




### 2. Testing Baseline_1_Unimodal_Video_Xception and Baseline_1_Unimodal_Audio_Xception

First, ensure you have a [`wandb`](https://www.wandb.com/) account as this experiment logs results using wandb.

#### Steps 1: Set Up Wandb Account
Create a wandb account and get your API key.

#### Steps 2: Configure the Experiment
- Open the `.../FakeMix/run.sh` file.
- Set the `--model` parameter to the directory name in `.../FakeMix/models` where you want to run the experiment. For example:
  ```bash
  --model Baseline_1_Unimodal_Audio_Xception
  ```
- Set the `--mode` parameter to `test`. For example:
  ```bash
  --mode test
  ```
- If you want to use FakeMix for training, set the `--mode` parameter to `train`.

#### Steps 3: Update Configurations
- In the experiment directory, open the `configs.yaml` file. For example:
  ```bash
  .../FakeMix/models/Baseline_1_Unimodal_Audio_Xception/configs.yaml
  ```
- Update the `wandb-login-key` value with your wandb login API key.
- Update the `wandb-init-entity` value with your wandb profile name or team name.
- Set the `wandb-init-config-dataset-data_path` value to the root path of the FakeMix dataset. For example:
  ```yaml
  data_path: "/home/lsy/laboratory/Research/FakeMix/data"
  ```
- Set the `wandb-init-config-engine-gpuid` value to the GPU ID you want to use for the experiment. For example, to use GPU 0:
  ```yaml
  gpuid: "0"
  ```

#### Steps 4: Run the Experiment
- Execute the experiment by running:
  ```bash
  sh .../FakeMix/run.sh
  ```

#### Steps 5. Check Results
- After the test is completed, the output will display `Average each accuracy` and `Total accuracy`. These values represent the TA and FDM metrics, respectively.
- Each experiment directory will also have an `each_file_record.xlsx` file, which shows the accuracy for each video clip, indicating which clips were correctly classified.

By following these steps, you can test the Baseline_1_Unimodal_Video_Xception and Baseline_1_Unimodal_Audio_Xception models and analyze the results using wandb and the provided Excel files.
<br>

### 3. Testing Baseline_2_Multimodal_AVAD

#### Step 1: Download AVAD Checkpoint
- Download Audio-visual synchronization model checkpoint `sync_model.pth`[link](https://drive.google.com/file/d/1BxaPiZmpiOJDsbbq8ZIDHJU7--RJE7Br/view?usp=sharing) at .../FakeMix/models/Baseline_2_Multimodal_AVAD/ 

#### Step 2: Generate Test Data Paths File
- Open `.../FakeMix/models/Baseline_2_Multimodal_AVAD/make_data_path_to_textfile.py`.
- Set the `root_directory` variable to the path of the FakeMix test directory.
- Set the `output_file` variable to the desired path where you want to save the `.txt` file.
- Run the following command to create a `.txt` file containing paths to the test data:
  ```bash
  python .../FakeMix/models/Baseline_2_Multimodal_AVAD/make_data_path_to_textfile.py
  ```

#### Step 3: Run Detection
- Navigate to the model directory:
  ```bash
  cd .../FakeMix/models/Baseline_2_Multimodal_AVAD
  ```
- Run the detection command:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python detect.py --test_video_path "/home/lsy/laboratory/Research/FakeMix/models/Baseline_2_Multimodal_AVAD/tools_for_FakeMix/FakeMIx_mp4_paths.txt" --device cuda:0 --max-len 50 --n_workers 18 --bs 1 --lam 0 --output_dir /home/lsy/laboratory/Research/FakeMix/models/Baseline_2_Multimodal_AVAD
  ```
- Set the command arguments as follows:
  - `CUDA_VISIBLE_DEVICES=n`: Set `n` to the GPU ID you want to use.
  - `test_video_path`: Path to the `.txt` file created in the previous step.
  - `device`: Set to the same value as `CUDA_VISIBLE_DEVICES`.
  - `max-len`: Maximum sequence length considered by AVAD. Set to 50 for this experiment.
  - `n_workers`: Number of CPU workers for the dataloader. Adjust based on your CPU specifications.
  - `bs`: Batch size. Set to 1 for testing.
  - `lam`: Fixed value.
  - `output_dir`: Path where the output results will be saved (e.g., `.../FakeMix/models/Baseline_2_Multimodal_AVAD`).

#### Step 4: Evaluate Results
- After the process completes, a `testing_scores.json` file will be created in `.../FakeMix/models/Baseline_2_Multimodal_AVAD`. This JSON file contains evaluation results for each test video and audio clip, including probability scores for detecting deepfakes per second.
- Open `.../FakeMix/models/Baseline_2_Multimodal_AVAD/calculate_our_metric.py`.
- Set the `file_path` variable to the path of the `testing_scores.json` file.
- Run the following command to calculate the metrics:
  ```bash
  python .../FakeMix/models/Baseline_2_Multimodal_AVAD/calculate_our_metric.py
  ```
- Once executed, you will obtain the TA and FDM evaluation results.
- Additionally, in the directory specified by the `file_path` variable, you will find neatly recorded JSON files detailing the evaluation results for each video clip.

By following these steps, you can effectively test the Baseline_2_Multimodal_AVAD model and analyze the results.


## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{jung2024and,
  title={WWW: Where, Which and Whatever Enhancing Interpretability in Multimodal Deepfake Detection},
  author={Jung, Juho and Lee, Sangyoun and Kang, Jooeon and Na, Yunjin},
  journal={arXiv preprint arXiv:2408.02954},
  year={2024}
}
