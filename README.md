# 企业文档 RAG 问答引擎（doc-rag-engine）

> 一个**企业文档 RAG** 问答系统，核心亮点是 **答案可溯源到 PDF 页码**：每个文本块在构建索引时记录其来源页码，
> 回答后会一并打印「来源页码」，方便人工核验。
> 可选接入 **Query 改写** 模块，把多轮对话里的指代 / 省略 / 对比 / 反问改写成适合检索的独立Query。

> 作者：[@MiyamotoXC](https://github.com/MiyamotoXC) ｜ 许可证：MIT ｜ 欢迎 Star / Fork 二次开发。

## 一、特性

- **页码溯源**：基于字符级页码映射，每个 chunk 标注其所属 PDF 页码，回答时回显来源页码
- **本地向量库**：FAISS（`IndexFlatL2`）本地落盘，支持先建库后加载，免去重复 Embedding
- **PDF / TXT 双输入**：`PyPDF2` 解析 PDF 并记录页码；纯文本直接读取
- **Query 改写增强**：`rewrite.py` 提供上下文依赖 / 对比 / 模糊指代 / 多意图 / 反问 五类改写与自动识别
- **密钥走环境变量**：`DASHSCOPE_API_KEY` 不入库

## 二、目录结构

```
doc-rag-engine/
├── config.py          # 统一配置（密钥走环境变量）
├── rag.py             # 解析 / 切分 / 向量化 / 页码溯源 / 检索问答
├── rewrite.py         # Query 改写器（五类改写 + 自动识别）
├── example_kb/        # 脱敏示例知识库（.txt）
├── requirements.txt
└── LICENSE
```

## 三、快速开始

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="你的密钥"        # Windows: set DASHSCOPE_API_KEY=...

# 构建并查询示例知识库
python rag.py

# 体验 Query 改写
python rewrite.py
```

把 `config.KB_FILE` 指向你自己的企业 PDF / TXT 即可投产。

## 四、配置（config.py）

| 项 | 默认 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 环境变量 | Embedding 与 LLM 密钥（必需，不入库） |
| `EMBEDDING_MODEL` | `text-embedding-v1` | 向量模型 |
| `LLM_MODEL` | `deepseek-v3` | 生成答案的 LLM |
| `KB_FILE` | `example_kb/sample.txt` | 知识库文件（.pdf / .txt） |
| `BUILD_DIR` | `vector_db` | 索引落盘目录（已被 .gitignore 忽略） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | 切分参数 |

## 五、能力边界（诚实说明）

- 示例知识库为脱敏的通用制度摘要，仅作技术演示
- 真实 PDF 的表格 / 扫描件解析质量取决于 `PyPDF2`，复杂排版建议预处理为文本
- LLM 回答质量依赖于检索召回；长文档可调小 `CHUNK_OVERLAP` 或改用语义切分

## 六、License

代码以 MIT 发布。
