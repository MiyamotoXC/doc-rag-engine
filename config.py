# -*- coding: utf-8 -*-
"""企业文档 RAG 问答引擎：统一配置（密钥走环境变量，不入库）。"""
import os
import dashscope

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")
dashscope.api_key = DASHSCOPE_API_KEY

EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "deepseek-v3"          # 经 dashscope 兼容网关访问

KB_FILE = "example_kb/sample.txt"  # 支持 .pdf / .txt
BUILD_DIR = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
