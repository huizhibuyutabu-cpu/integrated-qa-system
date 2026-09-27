# 导入配置解析库
import configparser
# 导入路径操作库
import os

class Config():
    # 初始化配置，加载 config.ini 文件
    def __init__(self,config_file="../config.ini"):
        # 创建配置解析器
        self.config=configparser.ConfigParser()
        # 获取当前文件（config.py）所在目录（即 base/）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 项目根目录（即 integrated_qa_system/）
        self.PROJECT_ROOT = os.path.dirname(base_dir)
        # 构造 config.ini 的绝对路径
        config_path = os.path.join(base_dir, config_file)

        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config.read_file(f)
            """
            “使用 UTF-8 编码安全地打开配置文件，交给配置解析器读取，并在读取完毕后自动关闭文件。”
            这样做完美绕开了 Windows 默认 GBK 编码的坑，同时保证了代码的健壮性。
            之后你的 self.MYSQL_HOST、self.REDIS_PASSWORD 等属性才能顺利从文件中读取出来。😊
            """
        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST=self.config.get('mysql','host',fallback="localhost")
        # MySQL 用户名
        self.MYSQL_USER=self.config.get('mysql','user',fallback="root")
        # MySQL 密码
        self.MYSQL_PASSWORD=self.config.get('mysql','password',fallback="root")
        # MySQL 数据库名
        self.MYSQL_DATABASE=self.config.get('mysql','database',fallback="subjects_kg")

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST=self.config.get('redis','host',fallback="localhost")
        # Redis 端口
        self.REDIS_PORT=self.config.get('redis','port',fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD=self.config.get('redis','password',fallback="root")
        # Redis 数据库编号
        self.REDIS_DB=self.config.get('redis','db',fallback=0)

        # 日志文件路径：从配置文件读取相对路径，然后基于项目根目录拼接为绝对路径
        log_rel_path = self.config.get('logger', 'log_file', fallback='logs/app.log')
        self.LOG_FILE = os.path.join(self.PROJECT_ROOT, log_rel_path)  # <--- 这行必须有

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 端口
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='default')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone')
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback=["ai", "java", "test", "ops", "bigdata","tj"]))

        # AC 配置
        self.AC_PORT = self.config.getint('ac', 'port', fallback=18077)
        self.AC_HOST = self.config.get('ac', 'host', fallback="0.0.0.0")

        # Nacos 配置
        self.NACOS_SERVER_ADDR = self.config.get('nacos', 'server-addr', fallback='127.0.0.1:8848')
        self.NACOS_USERNAME = self.config.get('nacos', 'username', fallback='nacos')
        self.NACOS_PASSWORD = self.config.get('nacos', 'password', fallback='nacos')
        self.NACOS_DISCOVERY_NAME = self.config.get('nacos', 'discovery_name', fallback='nacos')
        self.NACOS_DISCOVERY_NAMESPACE = self.config.get('nacos', 'discovery_namespace', fallback='public')
        self.NACOS_DISCOVERY_GROUP = self.config.get('nacos', 'discovery_group', fallback='DEFAULT_GROUP')
        self.NACOS_DISCOVERY_IP = self.config.get('nacos', 'discovery_ip', fallback='127.0.0.1')


if __name__ == '__main__':
    conf=Config()
    print(conf.MYSQL_HOST)
    print(conf.CHUNK_OVERLAP)
    print(conf.VALID_SOURCES)
    print(type(conf.VALID_SOURCES))
    print(conf.DASHSCOPE_API_KEY)
