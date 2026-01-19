"""
Data loading and preprocessing module
"""
import json
from typing import List, Dict, Any


def load_meeting_data(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load meeting data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with meeting dates as keys and lists of speech records as values
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_public_comments_and_hearings(meeting_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract public comments and public hearings from meeting data
    
    Args:
        meeting_data: List of speech records for a single meeting
        
    Returns:
        Dictionary containing public_comments and public_hearings
    """
    public_comments = []
    public_hearings = []
    
    for item in meeting_data:
        if item.get('is_public_comment', 0) == 1:
            public_comments.append(item)
        if item.get('is_public_hearing', 0) == 1:
            public_hearings.append(item)
    
    return {
        'public_comments': public_comments,
        'public_hearings': public_hearings
    }


def prepare_texts_for_embedding(meeting_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Prepare text data for embedding
    
    Args:
        meeting_data: All meeting data
        
    Returns:
        List containing text and metadata
    """
    documents = []
    
    for meeting_date, items in meeting_data.items():
        for item in items:
            # Build document text
            text = item.get('text', '')
            if not text:
                continue
                
            # Build metadata
            metadata = {
                'meeting_date': meeting_date,
                'speaker': item.get('speaker', ''),
                'start': item.get('start', 0),
                'end': item.get('end', 0),
                'is_public_comment': item.get('is_public_comment', 0),
                'is_public_hearing': item.get('is_public_hearing', 0),
                'meeting_section': item.get('Meeting Section', ''),
                'speaker_role': item.get('Speaker Role', '')
            }
            
            documents.append({
                'text': text,
                'metadata': metadata
            })
    
    return documents
