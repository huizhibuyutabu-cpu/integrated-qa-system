# utils/preprocess.py
# 导入分词库
import jieba
import re
import sys, os

# =============================================================================
# 新增：自定义 jieba 缓存目录，避免写入 C 盘
# 原因：jieba 默认将缓存文件（jieba.cache）写入系统临时目录（如 C:\Users\...\Temp），
#       这会占用 C 盘空间，且在多项目环境下可能产生冲突。
# 解决方案：将缓存目录重定向到当前文件所在的目录（即 mysql_qa/utils/）下，
#           这样缓存文件会生成在 E:\py_base\integrated_qa_system\mysql_qa\utils\jieba.cache。
# =============================================================================
# 注意：此设置必须在首次调用 jieba.lcut() 之前执行，因此放在文件顶部。
# 获取当前文件所在目录（即 utils/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 直接使用当前目录作为缓存目录
cache_dir = current_dir
# 确保目录存在（一般已存在，但以防万一）
os.makedirs(cache_dir, exist_ok=True)
# 将 jieba 的临时目录指向当前目录
# 之后 jieba.cache 将生成在 utils/ 下，不再写入 C 盘
jieba.dt.tmp_dir = cache_dir
# =============================================================================

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
"""
__file__ 是当前文件的完整路径（如 E:/py_base/.../mysql_client.py）。
os.path.abspath() 获取绝对路径，
os.path.dirname() 获取所在目录。
最终 current_dir 就是当前脚本所在的文件夹路径。
"""
module_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)
#将项目根目录添加到 Python 的模块搜索路径首位，以便后续导入 base 等自定义模块。

# --- 新增：将 base 目录加入 sys.path ---
base_dir = os.path.join(project_root, 'base')
sys.path.insert(0, base_dir)
"""
构造 base 目录的路径并插入搜索路径，使得可以直接 from base import Config, logger。
"""
# 导入日志
from base import logger

def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


if __name__ == '__main__':
    text = "这是一个测试句子。"
    print(preprocess_text(text))