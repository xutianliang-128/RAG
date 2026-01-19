# Binary Classification Usage Guide

## Overview

The RAG system now supports **binary classification** for extracting public comments and public hearings. Each document is classified as either:
- **1** = Positive (is a public comment/hearing)
- **0** = Negative (is not a public comment/hearing)

## Main Methods

### 1. Classify Public Comments

```python
from rag import RAGSystem
from vector_store import VectorStore

# Initialize
vector_store = VectorStore(collection_name="meeting_documents")
rag = RAGSystem(vector_store)

# Classify documents
result = rag.classify_public_comments(n_results=20)

# Access results
print(f"Total documents: {result['total_documents']}")
print(f"Positive (is public comment): {result['positive_count']}")
print(f"Negative (not public comment): {result['negative_count']}")

# Access individual classifications
for cls in result['classifications']:
    print(f"Document: {cls['document_id']}")
    print(f"Prediction: {cls['prediction']}")  # 0 or 1
    print(f"Text: {cls['text']}")
    print(f"Metadata: {cls['metadata']}")
```

### 2. Classify Public Hearings

```python
result = rag.classify_public_hearings(n_results=20)

# Same structure as above
for cls in result['classifications']:
    if cls['prediction'] == 1:
        print(f"Found public hearing: {cls['text']}")
```

### 3. Filter by Meeting Date

```python
# Classify only documents from a specific meeting
result = rag.classify_public_comments(
    n_results=50,
    meeting_date="AA_01_09_23"
)
```

## Output Format

The classification result is a dictionary with the following structure:

```python
{
    'classifications': [
        {
            'document_id': 'doc_0',
            'text': 'Document text content...',
            'metadata': {
                'meeting_date': 'AA_01_09_23',
                'speaker': 'SPEAKER_11',
                'start': 129.462,
                'end': 137.087,
                ...
            },
            'prediction': 1,  # 0 or 1
            'task_type': 'public_comment'
        },
        ...
    ],
    'total_documents': 20,
    'positive_count': 5,
    'negative_count': 15
}
```

## How It Works

1. **Retrieval**: RAG retrieves relevant documents from the vector database
2. **Classification**: OpenAI classifies each document as 0 or 1
3. **Output**: Returns structured results with predictions for each document

## Example: Save Results to JSON

```python
import json

result = rag.classify_public_comments(n_results=20)

# Save to file
with open('classifications.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
```

## Batch Processing

Documents are processed in batches of 10 to avoid token limits. The system automatically handles:
- JSON parsing
- Error handling
- Fallback parsing if JSON is malformed
