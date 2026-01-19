"""
Main program entry point
Classifies each utterance as Public Comment, Public Hearing, or Neither
"""
import os
import json
from typing import Dict, Any
from data_loader import load_meeting_data, prepare_texts_for_embedding
from vector_store import VectorStore
from utterance_classifier import UtteranceClassifier


def main():
    """Main function"""
    print("=" * 60)
    print("Meeting Transcript RAG System - Public Comments & Hearings Extraction")
    print("=" * 60)
    
    # 1. Load data
    print("\n[Step 1] Loading meeting data...")
    data_file = "AA_train.json"
    if not os.path.exists(data_file):
        print(f"Error: File not found {data_file}")
        return
    
    meeting_data = load_meeting_data(data_file)
    print(f"Successfully loaded {len(meeting_data)} meetings")
    
    # 2. Prepare documents
    print("\n[Step 2] Preparing documents...")
    documents = prepare_texts_for_embedding(meeting_data)
    print(f"Prepared {len(documents)} documents")
    
    # 3. Create vector store
    print("\n[Step 3] Creating vector database...")
    vector_store = VectorStore(collection_name="meeting_documents")
    
    # Check if data already exists
    existing_count = vector_store.get_collection_size()
    if existing_count == 0:
        print("Vector database is empty, adding documents...")
        vector_store.add_documents(documents)
    else:
        print(f"Vector database already contains {existing_count} documents, skipping add step")
        print("(To rebuild, delete the ./chroma_db directory)")
    
    # 4. Create utterance classifier
    print("\n[Step 4] Initializing utterance classifier...")
    try:
        classifier = UtteranceClassifier(vector_store)
        print("Utterance classifier initialized successfully")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure OPENAI_API_KEY is set in the .env file")
        return
    
    # 5. Prepare test utterances (using all utterances from training data as test)
    print("\n[Step 5] Preparing utterances for classification...")
    all_utterances = []
    utterance_id = 0
    
    for meeting_date, utterances in meeting_data.items():
        for utterance in utterances:
            text = utterance.get('text', '')
            if not text:
                continue
            
            all_utterances.append({
                'utterance_id': f"{meeting_date}_utt_{utterance_id}",
                'text': text,
                'metadata': {
                    'meeting_date': meeting_date,
                    'speaker': utterance.get('speaker', ''),
                    'start': utterance.get('start', 0),
                    'end': utterance.get('end', 0),
                    'ground_truth_comment': utterance.get('is_public_comment', 0),
                    'ground_truth_hearing': utterance.get('is_public_hearing', 0)
                }
            })
            utterance_id += 1
    
    print(f"Total utterances to classify: {len(all_utterances)}")
    
    # 6. Classify all utterances
    print("\n[Step 6] Classifying utterances...")
    print("=" * 60)
    classification_results = classifier.classify_utterances_batch(
        all_utterances,
        batch_size=20
    )
    
    # 7. Display results summary
    print("\n" + "=" * 60)
    print("Classification Results Summary")
    print("=" * 60)
    
    total = len(classification_results)
    public_comments = sum(1 for r in classification_results if r['is_public_comment'] == 1)
    public_hearings = sum(1 for r in classification_results if r['is_public_hearing'] == 1)
    neither = sum(1 for r in classification_results if r['is_public_comment'] == 0 and r['is_public_hearing'] == 0)
    
    print(f"\nTotal utterances: {total}")
    print(f"Public Comments: {public_comments}")
    print(f"Public Hearings: {public_hearings}")
    print(f"Neither: {neither}")
    
    # 8. Show sample results
    print("\n" + "-" * 60)
    print("Sample Results (first 10):")
    print("-" * 60)
    for i, result in enumerate(classification_results[:10]):
        comment_label = "Public Comment" if result['is_public_comment'] == 1 else "Not Public Comment"
        hearing_label = "Public Hearing" if result['is_public_hearing'] == 1 else "Not Public Hearing"
        
        print(f"\nUtterance {i+1} ({result['utterance_id']}):")
        print(f"  Text: {result['text'][:80]}...")
        print(f"  Prediction - {comment_label}, {hearing_label}")
        if 'ground_truth_comment' in result.get('metadata', {}):
            gt_comment = result['metadata']['ground_truth_comment']
            gt_hearing = result['metadata']['ground_truth_hearing']
            print(f"  Ground Truth - Comment: {gt_comment}, Hearing: {gt_hearing}")
    
    # 9. Save results to JSON
    print("\n" + "=" * 60)
    print("Saving results to JSON...")
    output_file = "utterance_classifications.json"
    
    # Format output as a list of classifications
    output_list = []
    for result in classification_results:
        output_list.append({
            'utterance_id': result['utterance_id'],
            'text': result['text'],
            'is_public_comment': result['is_public_comment'],
            'is_public_hearing': result['is_public_hearing'],
            'classification': _get_classification_label(result),
            'metadata': result.get('metadata', {})
        })
    
    output_data = {
        'total_utterances': total,
        'summary': {
            'public_comments': public_comments,
            'public_hearings': public_hearings,
            'neither': neither
        },
        'classifications': output_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print("=" * 60)
    
    # 10. Calculate accuracy if ground truth is available
    if any('ground_truth_comment' in r.get('metadata', {}) for r in classification_results):
        print("\n" + "=" * 60)
        print("Accuracy Metrics (if ground truth available)")
        print("=" * 60)
        
        correct_comment = 0
        correct_hearing = 0
        total_with_gt = 0
        
        for result in classification_results:
            metadata = result.get('metadata', {})
            if 'ground_truth_comment' in metadata:
                total_with_gt += 1
                if result['is_public_comment'] == metadata['ground_truth_comment']:
                    correct_comment += 1
                if result['is_public_hearing'] == metadata['ground_truth_hearing']:
                    correct_hearing += 1
        
        if total_with_gt > 0:
            comment_accuracy = correct_comment / total_with_gt
            hearing_accuracy = correct_hearing / total_with_gt
            print(f"\nPublic Comment Classification Accuracy: {comment_accuracy:.2%} ({correct_comment}/{total_with_gt})")
            print(f"Public Hearing Classification Accuracy: {hearing_accuracy:.2%} ({correct_hearing}/{total_with_gt})")
    
    print("\n" + "=" * 60)
    print("Classification completed!")
    print("=" * 60)


def _get_classification_label(result: Dict[str, Any]) -> str:
    """Get human-readable classification label"""
    if result['is_public_comment'] == 1 and result['is_public_hearing'] == 1:
        return "Public Comment and Public Hearing"
    elif result['is_public_comment'] == 1:
        return "Public Comment"
    elif result['is_public_hearing'] == 1:
        return "Public Hearing"
    else:
        return "Neither"
    
    print("\n" + "=" * 60)
    print("Query completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
