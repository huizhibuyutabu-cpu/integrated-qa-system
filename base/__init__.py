from .config import Config
from .logger import logger
# ★ 关键：创建全局唯一的配置实例
# 这样 `from base import config` 拿到的才是 Config() 实例，而不是模块
config = Config()
from .NacosConfig import nacos_config