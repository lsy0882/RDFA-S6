import os
import wandb
import torch
import inspect
import functools
import logging
import shutil

from omegaconf import DictConfig, OmegaConf
from loguru import logger


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        name = func.__name__
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry: logger_.log(level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs)
            result = func(*args, **kwargs)
            if exit: logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result
        return wrapped
    return wrapper

@logger_wraps()
def register_omegaconf_resolvers() -> None:
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("sub", lambda x, y: x - y)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
    OmegaConf.register_new_resolver("div", lambda x, y: x / y)
    OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
    OmegaConf.register_new_resolver("not", lambda x: not x)

@logger_wraps()
def configure_loguru(cfg: DictConfig):
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    for handler in cfg.loguru.handlers:
        if 'sink' in handler and handler['sink'].endswith('.log'):
            log_path = handler['sink'].format(time='{time}')
            logger.add(log_path, level=handler.level, format=handler.format)

@logger_wraps()
def wandb_setup(cfg: DictConfig) -> wandb.sdk.wandb_run.Run:
    os.environ["WANDB_IGNORE_GIT"] = "true"
    if not cfg.wandb.login.key:
        logger.warning("WandB login key is empty, aborting setup.")
        return None
    
    try:
        wandb.login(key=cfg.wandb.login.key)
    except Exception as e:
        logger.error(f"WandB login failed: {e}")
        return None
    
    try:
        return wandb.init(**cfg.wandb.init, config=OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        logger.error(f"WandB init failed: {e}")
        return None

@logger_wraps()
def log_model_information_to_wandb(wandb_run, model, input_shape, root_path):
    if not wandb_run:
        logger.error("Invalid wandb_run provided.")
        return
    
    try: # Log model architecture
        artifact_arch = wandb.Artifact("architecture", type="model", description="Architecture of the trained model", metadata={"framework": "pytorch"})
        with artifact_arch.new_file("model_arch.txt", mode="w") as f:
            for name, module in model.named_modules(): f.write(f"{name}: {type(module).__name__}\n")
        wandb_run.log_artifact(artifact_arch)
    except Exception as e: logger.error(f"Error in logging model architecture: {e}")
    
    try: # Convert and log model to ONNX
        onnx_file_name = "model.onnx"
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(model, dummy_input, onnx_file_name)
        artifact_onnx = wandb.Artifact("architecture", type="model", description="ONNX model file")
        artifact_onnx.add_file(onnx_file_name)
        wandb_run.log_artifact(artifact_onnx)
        logger.info(f"ONNX model saved and logged to wandb: {onnx_file_name}")
    except Exception as e: logger.error(f"Failed to save and log ONNX model: {e}")
    
    
    # try: # Log directory files
    #     def add_files_to_artifact(dir_path, artifact):
    #         for item in os.listdir(dir_path):
    #             item_path = os.path.join(dir_path, item)
    #             if os.path.isfile(item_path): artifact.add_file(item_path, name=item_path[len(dir_path)+1:])
    #             elif os.path.isdir(item_path): add_files_to_artifact(item_path, artifact)
    #     artifact_dirfiles = wandb.Artifact("files", type="data", description="All files and subdirectories from the specified directory")
    #     add_files_to_artifact(root_path, artifact_dirfiles)
    #     wandb_run.log_artifact(artifact_dirfiles)
    # except Exception as e: logger.error(f"Error in logging directory files: {e}")
    
    logger.debug(f"Complete {__name__}.{inspect.currentframe().f_code.co_name}")


@logger_wraps()
def logging_files_to_log(cfg: DictConfig) -> None:
    # inner function
    def copy_files(src, dst, exclude_dirs=[], additional_paths=[]):
        if not os.path.exists(dst):
            os.makedirs(dst)  # 목적지 디렉토리 생성
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                if item not in exclude_dirs:
                    if os.path.exists(d):
                        shutil.rmtree(d)  # 기존 디렉토리 삭제
                    shutil.copytree(s, d, ignore=shutil.ignore_patterns(*exclude_dirs))
            else:
                if os.path.exists(d):
                    os.remove(d)  # 기존 파일 삭제
                shutil.copy2(s, d)

        # 추가 경로에 있는 파일 및 디렉토리 복사
        for path in additional_paths:
            s = path
            d = os.path.join(dst, os.path.basename(path))
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)  # 기존 디렉토리 삭제
                shutil.copytree(s, d, ignore=shutil.ignore_patterns(*exclude_dirs))
            else:
                if os.path.exists(d):
                    os.remove(d)  # 기존 파일 삭제
                shutil.copy2(s, d)
    
    # main
    destination_dir = os.path.join(cfg.log_path, "model_files")
    source_dir = cfg.model_path
    exclude_dirs = ["logs", "data", "annotations"]
    additional_paths = ["./libs", 
                        "./tasks/temporal_action_localization/benchmarks",
                        "./build/mamba/mamba_ssm"
                        ]
    copy_files(source_dir, destination_dir, exclude_dirs, additional_paths)