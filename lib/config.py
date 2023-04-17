"""配置文件类型定义，以便类型检查
"""
from dataclasses import dataclass


@dataclass
class LogConfig:
    log_interval: int
    log_file: str


@dataclass
class TrainConfig:
    num_epochs: int
    lr: float
    batch_size: int
    num_workers: int
    use_validation: bool


@dataclass
class Dataset:
    dataset_path: str
    checkpoint_directory: str
    checkpoint_prefix: str


@dataclass
class Config:
    dataset: Dataset
    model_file: str
    preprocessing: str
    train: TrainConfig
    log: LogConfig
    plot: bool
