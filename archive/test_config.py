from dataclasses import dataclass, field

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class Config:
    dataset: str
    model_file: str
    train: dict = field(default_factory=dict)
    log: dict = field(default_factory=dict)
    plot: bool = False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    config = Config(**cfg)
    config = OmegaConf.structured(config)
    print(OmegaConf.to_yaml(config))
    dataset = config.dataset
    model_file = config.model_file
    num_epochs = config.train.num_epochs
    lr = config.train.lr
    batch_size = config.train.batch_size
    num_workers = config.train.num_workers
    use_validation = config.train.use_validation
    log_interval = config.log.log_interval
    log_file = config.log.log_file
    plot = config.plot
    print(config.log.log_interval)
    # 使用这些参数运行程序...


if __name__ == "__main__":
    my_app()
