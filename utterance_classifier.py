"""
Utterance-level binary classifier
Classifies each utterance as Public Comment, Public Hearing, or Neither
"""
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore

load_dotenv()


class UtteranceClassifier:
    """Classifier for individual utterances"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize utterance classifier
        
        Args:
            vector_store: Vector store instance with training data
        """
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
    
    def classify_utterance(self, 
                          utterance_text: str,
                          context_utterances: Optional[List[str]] = None,
                          model: str = "gpt-5.2") -> Dict[str, int]:
        """
        Classify a single utterance
        
        Args:
            utterance_text: Text of the utterance to classify
            context_utterances: Optional list of surrounding utterances for context
            model: OpenAI model name
            
        Returns:
            Dictionary with predictions: {'is_public_comment': 0/1, 'is_public_hearing': 0/1}
        """
        # Retrieve similar training examples from vector store
        similar_docs = self.vector_store.search(
            utterance_text,
            n_results=5
        )
        
        # Build context from similar examples
        examples_context = self._build_examples_context(similar_docs)
        
        # Build context from surrounding utterances if provided
        surrounding_context = ""
        if context_utterances:
            surrounding_context = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context_utterances)])
        
        # Build prompt
        prompt = self._build_classification_prompt(
            utterance_text,
            examples_context,
            surrounding_context
        )
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a binary classifier for meeting transcripts. For each utterance, classify it separately for two categories: (1) is_public_comment: 1 if it is a public comment (citizen comment during public comment period), 0 otherwise; (2) is_public_hearing: 1 if it is a public hearing (formal hearing process), 0 otherwise. An utterance can be BOTH a public comment AND a public hearing. Return ONLY a valid JSON object with these two fields."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            predictions = self._parse_classification(result_text)
            
            # Combine: if either comment or hearing is 1, then prediction is 1
            is_comment = predictions.get('is_public_comment', 0)
            is_hearing = predictions.get('is_public_hearing', 0)
            is_public = 1 if (is_comment == 1 or is_hearing == 1) else 0
            
            return {
                'is_public_comment': is_comment,
                'is_public_hearing': is_hearing,
                'is_public': is_public  # Combined prediction: 1 if comment OR hearing, 0 otherwise
            }
            
        except Exception as e:
            print(f"Error classifying utterance: {e}")
            return {
                'is_public_comment': 0,
                'is_public_hearing': 0,
                'is_public': 0,
                'error': str(e)
            }
    
    def classify_utterances_batch(self,
                                  utterances: List[Dict[str, Any]],
                                  batch_size: int = 20,
                                  model: str = "gpt-5.2") -> List[Dict[str, Any]]:
        """
        Classify a batch of utterances
        
        Args:
            utterances: List of utterance dictionaries with 'text' and optionally 'metadata'
            batch_size: Number of utterances to process in each batch
            model: OpenAI model name
            
        Returns:
            List of classification results, one per utterance
        """
        results = []
        
        for i in range(0, len(utterances), batch_size):
            batch = utterances[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(utterances)-1)//batch_size + 1} ({len(batch)} utterances)...")
            
            for j, utterance in enumerate(batch):
                utterance_text = utterance.get('text', '')
                if not utterance_text:
                    results.append({
                        'utterance_id': utterance.get('utterance_id', f'utterance_{i+j}'),
                        'text': utterance_text,
                        'is_public_comment': 0,
                        'is_public_hearing': 0,
                        'is_public': 0,
                        'error': 'Empty text',
                        'metadata': utterance.get('metadata', {})
                    })
                    continue
                
                # Get surrounding context if available
                context_utterances = None
                if 'context_before' in utterance:
                    context_utterances = utterance['context_before']
                elif i+j > 0 and i+j < len(utterances):
                    # Use previous and next utterances as context
                    context_utterances = [
                        utterances[i+j-1].get('text', '') if i+j > 0 else '',
                        utterances[i+j+1].get('text', '') if i+j < len(utterances)-1 else ''
                    ]
                
                # Classify
                prediction = self.classify_utterance(
                    utterance_text,
                    context_utterances,
                    model=model
                )
                
                # Combine with original utterance data
                result = {
                    'utterance_id': utterance.get('utterance_id', f'utterance_{i+j}'),
                    'text': utterance_text,
                    'is_public_comment': prediction.get('is_public_comment', 0),
                    'is_public_hearing': prediction.get('is_public_hearing', 0),
                    'is_public': prediction.get('is_public', 0),  # Combined: 1 if comment OR hearing
                    'metadata': utterance.get('metadata', {})
                }
                
                if 'error' in prediction:
                    result['error'] = prediction['error']
                
                results.append(result)
        
        return results
    
    def _build_examples_context(self, similar_docs: List[Dict[str, Any]]) -> str:
        """Build context from similar training examples"""
        if not similar_docs:
            return "No similar examples found."
        
        context_parts = []
        for i, doc in enumerate(similar_docs, 1):
            text = doc['text']
            metadata = doc['metadata']
            is_comment = metadata.get('is_public_comment', 0)
            is_hearing = metadata.get('is_public_hearing', 0)
            
            context_parts.append(
                f"Example {i}:\n"
                f"Text: {text}\n"
                f"Label - Public Comment: {is_comment}, Public Hearing: {is_hearing}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_classification_prompt(self,
                                     utterance_text: str,
                                     examples_context: str,
                                     surrounding_context: str = "") -> str:
        """Build prompt for utterance classification"""
        context_section = ""
        if surrounding_context:
            context_section = f"\n\nSurrounding Context:\n{surrounding_context}"
        
        return f"""Classify the following utterance from a meeting transcript.

Training Examples (with labels):
{examples_context}
{context_section}

Utterance to Classify:
{utterance_text}

Classify this utterance as:
1. is_public_comment: 1 if this is a public comment (citizen comment during public comment period), 0 otherwise
2. is_public_hearing: 1 if this is a public hearing (formal hearing process), 0 otherwise

Note: An utterance can be BOTH a public comment AND a public hearing. Classify each separately.

Return ONLY a JSON object in this format:
{{
  "is_public_comment": 0,
  "is_public_hearing": 0
}}"""
    
    def _parse_classification(self, text: str) -> Dict[str, int]:
        """Parse classification response from OpenAI"""
        try:
            # Remove markdown code blocks if present
            text_clean = text
            if '```json' in text:
                text_clean = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text_clean = text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(text_clean)
            
            # Ensure values are 0 or 1
            return {
                'is_public_comment': 1 if result.get('is_public_comment', 0) in [1, '1', True] else 0,
                'is_public_hearing': 1 if result.get('is_public_hearing', 0) in [1, '1', True] else 0
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: try to extract from text
            import re
            comment_match = re.search(r'is_public_comment["\']?\s*:\s*([01])', text, re.IGNORECASE)
            hearing_match = re.search(r'is_public_hearing["\']?\s*:\s*([01])', text, re.IGNORECASE)
            
            return {
                'is_public_comment': int(comment_match.group(1)) if comment_match else 0,
                'is_public_hearing': int(hearing_match.group(1)) if hearing_match else 0
            }
