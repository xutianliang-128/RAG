# Train-Test Pipeline Usage Guide

## Overview

This pipeline trains on multiple cities and tests on a specified city, automatically calculating evaluation metrics (accuracy, F1, precision, recall).

## File Structure

Expected file structure:
```
.
├── AA_train.json
├── AA_test.json
├── BB_train.json
├── BB_test.json
├── CC_train.json
├── CC_test.json
├── ...
└── train_test.py
```

## Usage

### Basic Usage

```bash
python train_test.py AA
```

This will:
- Train on cities: BB, CC, DD, EE, FF (all except AA)
- Test on: AA_test.json
- Calculate metrics and save results to `results_AA.json`

### Interactive Mode

```bash
python train_test.py
```

Then enter the test city when prompted.

### Programmatic Usage

```python
from train_test import TrainTestPipeline

# Define all cities
cities = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF']

# Create pipeline for testing on AA
pipeline = TrainTestPipeline(
    cities=cities,
    test_city='AA'
)

# Run pipeline
results = pipeline.run(batch_size=20)

# Access metrics
print(f"Public Comment F1: {results['evaluation']['public_comment']['f1']}")
print(f"Public Hearing F1: {results['evaluation']['public_hearing']['f1']}")
```

## Output

### Console Output

The script will display:
- Training progress
- Testing progress
- Evaluation metrics (accuracy, F1, precision, recall) for both tasks

### JSON Output

Results are saved to `results_{CITY}.json` with the following structure:

```json
{
  "test_city": "AA",
  "train_cities": ["BB", "CC", "DD", "EE", "FF"],
  "total_test_utterances": 1000,
  "evaluation": {
    "public_comment": {
      "accuracy": 0.95,
      "f1": 0.92,
      "precision": 0.94,
      "recall": 0.90,
      "support": 1000
    },
    "public_hearing": {
      "accuracy": 0.98,
      "f1": 0.85,
      "precision": 0.88,
      "recall": 0.82,
      "support": 1000
    }
  },
  "classifications": [
    {
      "utterance_id": "AA_01_09_23_utt_0",
      "text": "...",
      "is_public_comment": 1,
      "is_public_hearing": 0,
      "metadata": {...}
    },
    ...
  ]
}
```

## Configuration

### Modify City List

Edit `train_test.py` and update the `default_cities` list:

```python
default_cities = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF']  # Your actual cities
```

### Adjust Batch Size

```python
results = pipeline.run(batch_size=10)  # Smaller batches for slower processing
```

## Evaluation Metrics

The pipeline calculates:
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

Metrics are calculated separately for:
- Public Comment classification
- Public Hearing classification

## Notes

- Vector databases are cached by collection name (based on train cities)
- If a vector database already exists, you'll be prompted to rebuild
- To force rebuild, delete the `./chroma_db` directory
- Test data must have ground truth labels (`is_public_comment`, `is_public_hearing`)
