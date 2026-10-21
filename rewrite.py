# Query改写使用示例
# 导入依赖库
import dashscope
import os
import json
from http import HTTPStatus

from config import *

# 基于 prompt 生成文本
def get_completion(prompt, model="qwen-turbo-latest"):
    messages = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        temperature=0,
    )
    # 失败时 output 可能为 None（如未配置 API Key、HTTP 非 200），需先校验再取 choices
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"DashScope 调用失败: HTTP {response.status_code}, "
            f"code={response.code}, message={response.message}"
        )
    out = response.output
    if out is None or not getattr(out, "choices", None):
        raise RuntimeError(
            f"DashScope 未返回生成内容: code={response.code}, message={response.message}"
        )
    return out.choices[0].message.content

# Query改写功能
class QueryRewriter:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
    
    def rewrite_context_dependent_query(self, current_query, conversation_history):
        """上下文依赖型Query改写"""
        instruction = """
你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    
    def rewrite_comparative_query(self, query, context_info):
        """对比型Query改写"""
        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

### 改写后的查询 ###
"""
        
        return get_completion(prompt, self.model)
    
    def rewrite_ambiguous_reference_query(self, current_query, conversation_history):
        """模糊指代型Query改写"""
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个" 等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    
    def rewrite_multi_intent_query(self, query):
        """多意图型Query改写 - 分解查询"""
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以JSON数组格式输出。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 原始问题 ###
{query}

### 分解后的问题列表 ###
请以JSON数组格式输出，例如：["问题1", "问题2", "问题3"]
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return [response]
    
    def rewrite_rhetorical_query(self, current_query, conversation_history):
        """反问型Query改写"""
        instruction = """
你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        
        return get_completion(prompt, self.model)
    
    def auto_rewrite_query(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写"""
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回JSON格式的结果：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询",
    "confidence": "置信度(0-1)"
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 分析结果 ###
"""
        
        response = get_completion(prompt, self.model)
        try:
            return json.loads(response)
        except:
            return {
                "query_type": "未知类型",
                "rewritten_query": query,
                "confidence": 0.5
            }
    
    def auto_rewrite_and_execute(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写，然后根据类型调用相应的改写方法"""
        # 首先进行自动识别
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        
        # 根据识别结果调用相应的改写方法
        query_type = result.get('query_type', '')
        
        if '上下文依赖' in query_type:
            final_result = self.rewrite_context_dependent_query(query, conversation_history)
        elif '对比' in query_type:
            final_result = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif '模糊指代' in query_type:
            final_result = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif '多意图' in query_type:
            final_result = self.rewrite_multi_intent_query(query)
        elif '反问' in query_type:
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            # 对于其他类型，返回自动识别的改写结果
            final_result = result.get('rewritten_query', query)
        
        return {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get('confidence', 0.5),
            "rewritten_query": final_result,
            "auto_rewrite_result": result
        }

def main():
    rewriter = QueryRewriter()
    print("=== Query 改写示例（企业知识库问答） ===\n")
    cases = [
        ("上下文依赖", "还有其他报销材料吗？", "用户: 我想报销差旅费。\nAI: 需要行程单、发票和审批单。"),
        ("对比", "哪个方案更省成本？", "用户提到了方案A和方案B。"),
        ("模糊指代", "都什么时候开始？", "用户询问了上海和香港分支的年会。"),
        ("多意图", "门票多少钱？需要预约吗？停车费怎么收？", ""),
        ("反问", "这不会也要排两小时队吧？", "用户听说某项目很热门。"),
    ]
    for label, query, hist in cases:
        print(f"[{label}] 原始: {query}")
        result = rewriter.auto_rewrite_query(query, conversation_history=hist)
        print(f"  识别类型: {result.get('query_type')}")
        print(f"  改写结果: {result.get('rewritten_query')}\n")


if __name__ == "__main__":
    main() 