from cache.redis_client import RedisClient
r = RedisClient().client

# 删除 BM25 索引相关的缓存键
r.delete("qa_original_questions", "qa_tokenized_questions")

# 也可以同时删除所有 answer 缓存（可选）
keys = r.keys("answer:*")
if keys:
    r.delete(*keys)

print("已清除 BM25 索引缓存，重启后将自动从 MySQL 重建")