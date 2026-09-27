from nacos import NacosClient
from base import config


class NacosConfig:
    """
    Nacos 配置管理器。

    功能：
    1. 连接 Nacos 配置中心，获取配置信息
    2. 连接 Nacos 注册中心，实现服务发现
    3. 提供全局可访问的客户端实例
    """

    def __init__(self):
        # ---------------- 创建 Nacos 注册中心客户端 ----------------
        self.__discovery_client = NacosClient(
            server_addresses="http://" + config.NACOS_SERVER_ADDR,  # Nacos 服务器地址
            namespace=config.NACOS_DISCOVERY_NAMESPACE,            # 注册命名空间
            username=config.NACOS_USERNAME,                        # 用户名
            password=config.NACOS_PASSWORD,                         # 密码
            logDir="logs/",
        )

        # 关闭客户端缓存，保证每次获取最新数据
        self.__discovery_client.no_snapshot = True

    def get_discovery_client(self):
        """获取 Nacos 注册中心客户端实例"""
        return self.__discovery_client

    # ---------------- 服务注册信息 ----------------
    def get_discovery_ip(self):
        """获取本机服务 IP"""
        return config.NACOS_DISCOVERY_IP

    def get_discovery_name(self):
        """获取本机服务名称"""
        return config.NACOS_DISCOVERY_NAME

    def get_discovery_group(self):
        """获取注册服务分组"""
        return config.NACOS_DISCOVERY_GROUP


# ---------------- 全局 Nacos 配置实例 ----------------
nacos_config = NacosConfig()