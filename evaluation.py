"""
Evaluation metrics for binary classification
Calculates accuracy, F1, precision, and recall
"""
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def calculate_metrics(y_true: List[int], y_pred: List[int], task_name: str = "Classification") -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: List of true labels (0 or 1)
        y_pred: List of predicted labels (0 or 1)
        task_name: Name of the task for reporting
        
    Returns:
        Dictionary with metrics: accuracy, f1, precision, recall
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }


def evaluate_classifications(results: List[Dict[str, Any]], 
                            combined_task: bool = True) -> Dict[str, Any]:
    """
    Evaluate classification results against ground truth
    Combined task: Public Comment OR Public Hearing = 1, Others = 0
    
    Args:
        results: List of classification results with 'is_public' (combined prediction)
                 and metadata containing ground truth
        combined_task: Whether to evaluate combined classification (comment OR hearing)
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluation = {}
    
    # Extract ground truth and predictions for combined task
    # Ground truth: 1 if comment OR hearing, 0 otherwise
    # Prediction: 1 if comment OR hearing, 0 otherwise
    y_true_combined = []
    y_pred_combined = []
    
    for result in results:
        metadata = result.get('metadata', {})
        if 'ground_truth_comment' in metadata and 'ground_truth_hearing' in metadata:
            # Ground truth: 1 if either comment or hearing is 1
            gt_comment = metadata['ground_truth_comment']
            gt_hearing = metadata['ground_truth_hearing']
            gt_combined = 1 if (gt_comment == 1 or gt_hearing == 1) else 0
            
            # Prediction: 1 if either comment or hearing is 1
            pred_comment = result.get('is_public_comment', 0)
            pred_hearing = result.get('is_public_hearing', 0)
            pred_combined = result.get('is_public', 0)
            # Fallback: calculate if not provided
            if 'is_public' not in result:
                pred_combined = 1 if (pred_comment == 1 or pred_hearing == 1) else 0
            
            y_true_combined.append(gt_combined)
            y_pred_combined.append(pred_combined)
    
    if len(y_true_combined) > 0:
        evaluation['combined'] = calculate_metrics(
            y_true_combined,
            y_pred_combined,
            "Public Comment or Hearing (Combined)"
        )
        evaluation['combined']['support'] = len(y_true_combined)
        
        # Also calculate per-class metrics
        positive_count = sum(y_true_combined)
        negative_count = len(y_true_combined) - positive_count
        evaluation['combined']['positive_count'] = positive_count
        evaluation['combined']['negative_count'] = negative_count
    else:
        evaluation['combined'] = {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'support': 0,
            'positive_count': 0,
            'negative_count': 0
        }
    
    return evaluation


def print_evaluation_report(evaluation: Dict[str, Any]):
    """Print formatted evaluation report"""
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    
    if 'combined' in evaluation:
        metrics = evaluation['combined']
        print("\nPUBLIC COMMENT OR HEARING (Combined Binary Classification):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if 'support' in metrics:
            print(f"  Support:  {metrics['support']}")
        if 'positive_count' in metrics:
            print(f"  Positive (Public Comment/Hearing): {metrics['positive_count']}")
            print(f"  Negative (Others): {metrics['negative_count']}")
