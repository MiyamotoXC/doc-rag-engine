from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import os
import pickle

from config import *

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[Tuple[str, int]]]:
    """
    从PDF中提取文本并记录每个字符对应的页码
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
    """
    text = ""
    char_page_mapping = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            # 为当前页面的每个字符记录页码
            char_page_mapping.extend([page_number] * len(extracted_text))
        else:
            print(f"No text found on page {page_number}.")

    return text, char_page_mapping


def extract_text_from_txt(path: str):
    """从纯文本文件读取内容（整篇视为第 1 页，页码溯源仍可用）。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text, [1] * len(text)


def process_text_with_splitter(text: str, char_page_mapping: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 为每个文本块找到对应的页码信息
    page_info = {}
    current_pos = 0
    
    for chunk in chunks:
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)
        
        # 找到这个文本块中字符对应的页码
        chunk_pages = char_page_mapping[chunk_start:chunk_end]
        
        # 取页码的众数（出现最多的页码）作为该块的页码
        if chunk_pages:
            # 统计每个页码出现的次数
            page_counts = {}
            for page in chunk_pages:
                page_counts[page] = page_counts.get(page, 0) + 1
            
            # 找到出现次数最多的页码
            most_common_page = max(page_counts, key=page_counts.get)
            page_info[chunk] = most_common_page
        else:
            page_info[chunk] = 1  # 默认页码
        
        current_pos = chunk_end
    
    knowledgeBase.page_info = page_info
    print(f'页码映射完成，共 {len(page_info)} 个文本块')
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")
    
    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
    
    return knowledgeBase

def build_from_file(path: str, save_path: str = None):
    """从 PDF 或 TXT 解析文本并构建知识库（含页码溯源）。"""
    if path.lower().endswith('.pdf'):
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        text, mapping = extract_text_with_page_numbers(reader)
    else:
        text, mapping = extract_text_from_txt(path)
    print(f"提取的文本长度: {len(text)} 个字符。")
    return process_text_with_splitter(text, mapping, save_path=save_path)


def ask(knowledge_base, query: str, model_name: str = None) -> str:
    """检索 Top-K 文本块拼成上下文交给 LLM 生成答案，并打印来源页码。"""
    from langchain_community.llms import Tongyi
    llm = Tongyi(model_name=model_name or LLM_MODEL, dashscope_api_key=DASHSCOPE_API_KEY)
    docs = knowledge_base.similarity_search(query, k=10)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"根据以下上下文回答问题:\n\n{context}\n\n问题: {query}"
    response = llm.invoke(prompt)
    print(response)
    unique_pages = sorted({knowledge_base.page_info.get(d.page_content.strip(), '未知') for d in docs})
    print("来源页码:", unique_pages)
    return response


if __name__ == "__main__":
    kb = build_from_file(KB_FILE, save_path=BUILD_DIR)
    ask(kb, "请概述这份知识库的主要内容和关键条款")

