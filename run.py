import hydra
import importlib
from omegaconf import DictConfig, OmegaConf
from libs.utils import util_system

@hydra.main(config_path=".", config_name="run", version_base=None)
def main(run_cfg: DictConfig) -> None:
    
    # Register OmegaConf resolvers (add, sub, mul, div, floordiv)
    util_system.register_omegaconf_resolvers() 
    
    # Set custom job name
    util_system.configure_loguru(run_cfg)
    
    # Load additional configs & Merge
    engine_cfg = OmegaConf.load(f"{run_cfg.model_path}/engine.yaml")
    dataset_cfg = OmegaConf.load(f"{run_cfg.benchmark_path}/dataset.yaml")
    model_cfg = OmegaConf.load(f"{run_cfg.model_path}/model.yaml")
    cfg = OmegaConf.merge(run_cfg, engine_cfg, dataset_cfg, model_cfg)
    
    # Load target model's module
    target_module = importlib.import_module(cfg.model_path.replace("/", ".") + ".engine")
    target_engine = target_module.Engine(cfg)
    target_engine.run()

if __name__ == "__main__":
    main()