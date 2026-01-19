"""
RAG query module
Uses OpenAI API for generation and binary classification
"""
import os
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from vector_store import VectorStore

# Load environment variables
load_dotenv()


class RAGSystem:
    """RAG system class"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG system
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
    
    def query(self, 
              query_text: str, 
              n_results: int = 5,
              filter_dict: Optional[Dict] = None,
              model: str = "gpt-5.2") -> Dict[str, Any]:
        """
        Execute RAG query
        
        Args:
            query_text: Query text
            n_results: Number of documents to retrieve
            filter_dict: Optional filter conditions
            model: OpenAI model name
            
        Returns:
            Dictionary containing retrieved results and generated response
        """
        # 1. Retrieve relevant documents
        retrieved_docs = self.vector_store.search(
            query_text, 
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        # 2. Build context
        context = self._build_context(retrieved_docs)
        
        # 3. Build prompt
        prompt = self._build_prompt(query_text, context)
        
        # 4. Call OpenAI API
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional meeting transcript analyst, skilled at extracting public comments and public hearings from meeting records."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        generated_text = response.choices[0].message.content
        
        return {
            'query': query_text,
            'retrieved_documents': retrieved_docs,
            'generated_response': generated_text,
            'context_used': context
        }
    
    def classify_public_comments(self, 
                                 query_text: Optional[str] = None,
                                 n_results: int = 20,
                                 meeting_date: Optional[str] = None,
                                 model: str = "gpt-5.2") -> Dict[str, Any]:
        """
        Binary classification: Classify each document as public comment (1) or not (0)
        
        Args:
            query_text: Optional query text for retrieval
            n_results: Number of documents to retrieve and classify
            meeting_date: Optional meeting date filter
            model: OpenAI model name
            
        Returns:
            Dictionary containing classification results for each document
        """
        # Retrieve documents (without filtering by label)
        if query_text is None:
            query_text = "public comment or public hearing"
        
        filter_dict = {}
        if meeting_date:
            filter_dict['meeting_date'] = meeting_date
        
        retrieved_docs = self.vector_store.search(
            query_text,
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        if not retrieved_docs:
            return {
                'classifications': [],
                'total_documents': 0,
                'positive_count': 0
            }
        
        # Classify each document
        classifications = self._classify_documents(
            retrieved_docs,
            task_type="public_comment",
            model=model
        )
        
        positive_count = sum(1 for c in classifications if c['prediction'] == 1)
        
        return {
            'classifications': classifications,
            'total_documents': len(classifications),
            'positive_count': positive_count,
            'negative_count': len(classifications) - positive_count
        }
    
    def classify_public_hearings(self,
                                  query_text: Optional[str] = None,
                                  n_results: int = 20,
                                  meeting_date: Optional[str] = None,
                                  model: str = "gpt-5.2") -> Dict[str, Any]:
        """
        Binary classification: Classify each document as public hearing (1) or not (0)
        
        Args:
            query_text: Optional query text for retrieval
            n_results: Number of documents to retrieve and classify
            meeting_date: Optional meeting date filter
            model: OpenAI model name
            
        Returns:
            Dictionary containing classification results for each document
        """
        # Retrieve documents (without filtering by label)
        if query_text is None:
            query_text = "public hearing or public comment"
        
        filter_dict = {}
        if meeting_date:
            filter_dict['meeting_date'] = meeting_date
        
        retrieved_docs = self.vector_store.search(
            query_text,
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        if not retrieved_docs:
            return {
                'classifications': [],
                'total_documents': 0,
                'positive_count': 0
            }
        
        # Classify each document
        classifications = self._classify_documents(
            retrieved_docs,
            task_type="public_hearing",
            model=model
        )
        
        positive_count = sum(1 for c in classifications if c['prediction'] == 1)
        
        return {
            'classifications': classifications,
            'total_documents': len(classifications),
            'positive_count': positive_count,
            'negative_count': len(classifications) - positive_count
        }
    
    def _classify_documents(self, 
                           documents: List[Dict[str, Any]],
                           task_type: str = "public_comment",
                           model: str = "gpt-5.2") -> List[Dict[str, Any]]:
        """
        Classify documents using OpenAI for binary classification
        
        Args:
            documents: List of documents to classify
            task_type: "public_comment" or "public_hearing"
            model: OpenAI model name
            
        Returns:
            List of classification results
        """
        task_description = {
            "public_comment": "a public comment (citizen comment during public comment period)",
            "public_hearing": "a public hearing (formal hearing process)"
        }
        
        classifications = []
        
        # Process documents in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Build prompt for batch classification
            prompt = self._build_classification_prompt(batch, task_type, task_description[task_type])
            
            # Call OpenAI API
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are a binary classifier. Classify each document as 1 if it is {task_description[task_type]}, or 0 if it is not. Return ONLY a valid JSON object with a 'classifications' array."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1  # Lower temperature for more consistent classification
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Parse JSON response
                batch_results = self._parse_classification_response(result_text, len(batch))
                
                # Combine with document metadata
                for j, doc in enumerate(batch):
                    if j < len(batch_results):
                        prediction = batch_results[j]
                    else:
                        prediction = 0
                    
                    classifications.append({
                        'document_id': f"doc_{i+j}",
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'prediction': int(prediction),
                        'task_type': task_type
                    })
                    
            except Exception as e:
                print(f"Error classifying batch {i//batch_size + 1}: {e}")
                # Default to 0 if classification fails
                for j, doc in enumerate(batch):
                    classifications.append({
                        'document_id': f"doc_{i+j}",
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'prediction': 0,
                        'task_type': task_type,
                        'error': str(e)
                    })
        
        return classifications
    
    def _build_classification_prompt(self, 
                                     documents: List[Dict[str, Any]],
                                     task_type: str,
                                     task_description: str) -> str:
        """Build prompt for binary classification"""
        docs_text = []
        for i, doc in enumerate(documents):
            docs_text.append(
                f"Document {i+1}:\n"
                f"Speaker: {doc['metadata'].get('speaker', 'N/A')}\n"
                f"Meeting Date: {doc['metadata'].get('meeting_date', 'N/A')}\n"
                f"Text: {doc['text']}\n"
            )
        
        return f"""Classify each of the following documents as 1 if it is {task_description}, or 0 if it is not.

{docs_text}

Return a JSON array with predictions for each document in order. Format:
{{
  "classifications": [
    {{"document": 1, "prediction": 0}},
    {{"document": 2, "prediction": 1}},
    ...
  ]
}}

Only return the JSON, no additional text."""
    
    def _parse_classification_response(self, text: str, expected_count: int) -> List[int]:
        """Parse classification response from OpenAI"""
        predictions = []
        
        # Try to parse as JSON first
        try:
            # Remove markdown code blocks if present
            text_clean = text
            if '```json' in text:
                text_clean = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text_clean = text.split('```')[1].split('```')[0].strip()
            
            result_json = json.loads(text_clean)
            
            # Handle different JSON structures
            if isinstance(result_json, dict):
                if 'classifications' in result_json:
                    # Format: {"classifications": [{"document": 1, "prediction": 0}, ...]}
                    for item in result_json['classifications']:
                        if isinstance(item, dict):
                            pred = item.get('prediction', 0)
                        else:
                            pred = item
                        predictions.append(int(pred) if pred in [0, 1, "0", "1"] else 0)
                elif 'predictions' in result_json:
                    predictions = [int(p) if p in [0, 1, "0", "1"] else 0 for p in result_json['predictions']]
                else:
                    # Try to extract from values
                    for key, value in result_json.items():
                        if isinstance(value, (int, str)) and str(value) in ['0', '1']:
                            predictions.append(int(value))
            elif isinstance(result_json, list):
                # Format: [0, 1, 0, ...] or [{"prediction": 0}, ...]
                for item in result_json:
                    if isinstance(item, dict):
                        pred = item.get('prediction', 0)
                    else:
                        pred = item
                    predictions.append(int(pred) if pred in [0, 1, "0", "1"] else 0)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: extract numbers from text
            import re
            numbers = re.findall(r'\b[01]\b', text)
            for num in numbers[:expected_count]:
                predictions.append(int(num))
        
        # Pad with zeros if needed
        while len(predictions) < expected_count:
            predictions.append(0)
        
        return predictions[:expected_count]
    
    def extract_public_comments(self, meeting_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract public comments (legacy method, uses classification)
        
        Args:
            meeting_date: Optional meeting date filter
            
        Returns:
            Classification results
        """
        return self.classify_public_comments(meeting_date=meeting_date)
    
    def extract_public_hearings(self, meeting_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract public hearings (legacy method, uses classification)
        
        Args:
            meeting_date: Optional meeting date filter
            
        Returns:
            Classification results
        """
        return self.classify_public_hearings(meeting_date=meeting_date)
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc['text']
            metadata = doc['metadata']
            context_parts.append(
                f"[Document {i}]\n"
                f"Speaker: {metadata.get('speaker', 'N/A')}\n"
                f"Meeting Date: {metadata.get('meeting_date', 'N/A')}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt"""
        return f"""Based on the following meeting transcript context, answer the user's question.

Context Information:
{context}

User Question: {query}

Please provide an accurate and detailed answer based on the context information. If there is no relevant information in the context, please state so clearly."""
