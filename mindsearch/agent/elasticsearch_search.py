# mindsearch/agent/elasticsearch_search.py

from elasticsearch import Elasticsearch
from lagent.actions import BaseAction

class ElasticsearchSearch(BaseAction):
    def __init__(self, es_host, es_port, es_index, **kwargs):
        super().__init__(**kwargs)
        # 将 es_port 转换为整数
        es_port = int(es_port)
        self.es = Elasticsearch(
            [{'host': es_host, 'port': es_port, 'scheme': 'http'}]  # 添加 scheme 参数
        )
        self.es_index = es_index

    def run(self, query):
        # 构建搜索请求
        search_body = {
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        # 执行搜索
        response = self.es.search(index=self.es_index, body=search_body)
        # 提取搜索结果
        results = [hit['_source'] for hit in response['hits']['hits']]
        return results