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
    name: str
    dataset_path: str
    checkpoint_directory: str
    checkpoint_prefix: str


@dataclass
class ExtractorConfig:
    name: str
    num_kpt: int
    resize: list
    det_th: float


@dataclass
class MatcherConfig:
    name: str
    model_dir: str
    seed_top_k: list
    seed_radius_coe: float
    net_channels: int
    layer_num: int
    head: int
    seedlayer: list
    use_mc_seeding: bool
    use_score_encoding: bool
    conf_bar: list
    sink_iter: list
    detach_iter: int
    p_th: float


@dataclass
class RansacConfig:
    name: str


@dataclass
class MethodConfig:
    extractor: ExtractorConfig
    matcher: MatcherConfig
    ransac: RansacConfig
    dataset: Dataset
    model_file: str
    preprocessing: str
    train: TrainConfig
    log: LogConfig
    plot: bool


@dataclass
class Config:
    method: MethodConfig
    dataset: Dataset
    model_file: str
    preprocessing: str
    train: TrainConfig
    log: LogConfig
    plot: bool
