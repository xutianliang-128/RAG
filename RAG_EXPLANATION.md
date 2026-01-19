# RAG 工作原理说明

## RAG 与 OpenAI 的协作流程

### 整体架构

```
用户查询
    ↓
[步骤 1: Retrieval 检索]
    ↓
向量数据库 (ChromaDB) → 找到最相关的文档
    ↓
[步骤 2: Augmentation 增强]
    ↓
将检索到的文档作为上下文
    ↓
[步骤 3: Generation 生成]
    ↓
OpenAI API → 基于上下文生成回答
    ↓
最终答案
```

## 详细步骤解析

### 步骤 1: Retrieval（检索阶段）

**RAG 的角色：智能搜索引擎**

当用户提问时，RAG 系统会：
1. 将用户查询转换为向量（embedding）
2. 在向量数据库中搜索最相似的文档
3. 返回最相关的几个文档片段

```python
# 在 rag.py 的 query 方法中
retrieved_docs = self.vector_store.search(
    query_text,  # 用户的问题
    n_results=5,  # 返回最相关的 5 个文档
    filter_dict={'is_public_comment': 1}  # 可选过滤条件
)
```

**为什么需要这一步？**
- OpenAI 模型本身不知道你的会议数据
- 直接问 OpenAI 关于你的会议内容，它会"编造"答案
- RAG 先找到真实的相关文档，再让 OpenAI 基于这些文档回答

### 步骤 2: Augmentation（增强阶段）

**RAG 的角色：上下文构建器**

将检索到的文档组织成结构化的上下文：

```python
# 在 rag.py 的 _build_context 方法中
context = """
[Document 1]
Speaker: SPEAKER_11
Meeting Date: AA_01_09_23
Content: Good evening and welcome to the January 9 meeting...

[Document 2]
Speaker: SPEAKER_15
Meeting Date: AA_01_09_23
Content: I would like to make a public comment about...
"""
```

### 步骤 3: Generation（生成阶段）

**OpenAI 的角色：智能文本生成器**

将上下文和问题一起发送给 OpenAI：

```python
prompt = f"""
Based on the following meeting transcript context, answer the user's question.

Context Information:
{context}  # ← RAG 提供的真实文档

User Question: {query}  # ← 用户的问题

Please provide an accurate and detailed answer based on the context information.
"""

response = self.client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a professional meeting transcript analyst..."},
        {"role": "user", "content": prompt}
    ]
)
```

## RAG 扮演的角色

### 1. **知识库管理员**
- 存储和管理你的会议数据
- 将文本转换为可搜索的向量形式
- 快速找到相关文档

### 2. **信息筛选器**
- 从大量数据中筛选出最相关的部分
- 避免将整个数据集都发送给 OpenAI（节省 token 和成本）
- 只提供最相关的上下文

### 3. **事实提供者**
- 确保 OpenAI 基于真实数据回答，而不是"编造"
- 提供可追溯的文档来源
- 提高答案的准确性和可信度

## 对比：没有 RAG vs 有 RAG

### 没有 RAG（直接问 OpenAI）
```
用户: "这个会议中提到了哪些 public comments？"
OpenAI: "根据我的知识，public comments 通常包括..." 
        ❌ 这是通用知识，不是你的实际会议内容
```

### 有 RAG
```
用户: "这个会议中提到了哪些 public comments？"
RAG: 先检索你的会议数据，找到相关的 public comments 文档
OpenAI: 基于 RAG 提供的真实文档，总结出实际的 public comments
        ✅ 这是基于你的真实数据
```

## 代码中的具体实现

### 检索部分（vector_store.py）
```python
# 使用 sentence-transformers 将查询转换为向量
query_embedding = self.embedding_model.encode([query])

# 在 ChromaDB 中搜索相似文档
results = self.collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

### 增强部分（rag.py）
```python
# 将检索到的文档组织成上下文
context = self._build_context(retrieved_docs)
```

### 生成部分（rag.py）
```python
# 将上下文和问题一起发送给 OpenAI
prompt = self._build_prompt(query_text, context)
response = self.client.chat.completions.create(...)
```

## 优势总结

1. **准确性**：基于真实数据，不是模型的知识库
2. **效率**：只检索相关文档，节省 token
3. **可追溯**：可以查看答案来自哪些文档
4. **灵活性**：可以轻松更新知识库，无需重新训练模型
5. **成本**：只发送相关上下文，降低 API 调用成本
