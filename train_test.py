"""
Automated training and testing script
Trains on multiple cities and tests on a specified city
"""
import os
import json
import shutil
from typing import List, Dict, Any, Optional
from data_loader import load_meeting_data, prepare_texts_for_embedding
from vector_store import VectorStore
from utterance_classifier import UtteranceClassifier
from evaluation import evaluate_classifications, print_evaluation_report


class TrainTestPipeline:
    """Pipeline for training on multiple cities and testing on one city"""
    
    def __init__(self, cities: List[str], test_city: str, data_dir: str = "."):
        """
        Initialize pipeline
        
        Args:
            cities: List of all city codes (e.g., ['AA', 'BB', 'CC', ...])
            test_city: City code to use for testing
            data_dir: Directory containing data files
        """
        self.cities = cities
        self.test_city = test_city
        self.data_dir = data_dir
        self.train_cities = [c for c in cities if c != test_city]
        
        if test_city not in cities:
            raise ValueError(f"Test city {test_city} not in cities list: {cities}")
        
        print(f"Train cities: {self.train_cities}")
        print(f"Test city: {test_city}")
    
    def load_train_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load training data from all train cities"""
        all_train_data = {}
        
        for city in self.train_cities:
            train_file = os.path.join(self.data_dir, f"{city}_train.json")
            if not os.path.exists(train_file):
                print(f"Warning: Training file not found: {train_file}")
                continue
            
            print(f"Loading training data from {city}...")
            city_data = load_meeting_data(train_file)
            
            # Add city prefix to meeting dates to avoid conflicts
            for meeting_date, utterances in city_data.items():
                all_train_data[f"{city}_{meeting_date}"] = utterances
            
            print(f"  Loaded {len(city_data)} meetings from {city}")
        
        return all_train_data
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from test city"""
        test_file = os.path.join(self.data_dir, f"{self.test_city}_test.json")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        print(f"Loading test data from {self.test_city}...")
        test_data = load_meeting_data(test_file)
        
        # Convert to list of utterances
        test_utterances = []
        utterance_id = 0
        
        for meeting_date, utterances in test_data.items():
            for utterance in utterances:
                text = utterance.get('text', '')
                if not text:
                    continue
                
                test_utterances.append({
                    'utterance_id': f"{self.test_city}_{meeting_date}_utt_{utterance_id}",
                    'text': text,
                    'metadata': {
                        'meeting_date': meeting_date,
                        'speaker': utterance.get('speaker', ''),
                        'start': utterance.get('start', 0),
                        'end': utterance.get('end', 0),
                        'ground_truth_comment': utterance.get('is_public_comment', 0),
                        'ground_truth_hearing': utterance.get('is_public_hearing', 0),
                        'city': self.test_city
                    }
                })
                utterance_id += 1
        
        print(f"  Loaded {len(test_utterances)} test utterances")
        return test_utterances
    
    def train(self, collection_name: Optional[str] = None) -> VectorStore:
        """
        Train on all train cities
        
        Args:
            collection_name: Optional collection name (default: based on test city)
            
        Returns:
            Trained vector store
        """
        if collection_name is None:
            collection_name = f"train_{'_'.join(sorted(self.train_cities))}"
        
        print("\n" + "=" * 60)
        print("Training Phase")
        print("=" * 60)
        
        # Load training data
        train_data = self.load_train_data()
        
        if not train_data:
            raise ValueError("No training data loaded!")
        
        # Prepare documents for embedding
        print("\nPreparing documents for embedding...")
        documents = prepare_texts_for_embedding(train_data)
        print(f"Prepared {len(documents)} documents")
        
        # Create vector store
        print(f"\nCreating vector database (collection: {collection_name})...")
        vector_store = VectorStore(collection_name=collection_name)
        
        # Check if already exists
        existing_count = vector_store.get_collection_size()
        if existing_count > 0:
            print(f"Vector database already contains {existing_count} documents")
            response = input("Rebuild vector database? (y/n): ").strip().lower()
            if response == 'y':
                # Delete and rebuild
                if os.path.exists(vector_store.persist_directory):
                    shutil.rmtree(vector_store.persist_directory)
                vector_store = VectorStore(collection_name=collection_name)
                vector_store.add_documents(documents)
            else:
                print("Using existing vector database")
        else:
            vector_store.add_documents(documents)
        
        print(f"Training complete. Vector database contains {vector_store.get_collection_size()} documents")
        return vector_store
    
    def test(self, vector_store: VectorStore, batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        Test on test city
        
        Args:
            vector_store: Trained vector store
            batch_size: Batch size for classification
            
        Returns:
            List of classification results
        """
        print("\n" + "=" * 60)
        print("Testing Phase")
        print("=" * 60)
        
        # Load test data
        test_utterances = self.load_test_data()
        
        if not test_utterances:
            raise ValueError("No test data loaded!")
        
        # Create classifier
        print("\nInitializing classifier...")
        classifier = UtteranceClassifier(vector_store)
        
        # Classify test utterances
        print(f"\nClassifying {len(test_utterances)} test utterances...")
        classification_results = classifier.classify_utterances_batch(
            test_utterances,
            batch_size=batch_size
        )
        
        return classification_results
    
    def run(self, batch_size: int = 20, save_results: bool = True) -> Dict[str, Any]:
        """
        Run full pipeline: train and test
        
        Args:
            batch_size: Batch size for classification
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with evaluation results
        """
        # Train
        vector_store = self.train()
        
        # Test
        classification_results = self.test(vector_store, batch_size=batch_size)
        
        # Evaluate
        print("\n" + "=" * 60)
        print("Evaluation Phase")
        print("=" * 60)
        
        evaluation = evaluate_classifications(classification_results)
        print_evaluation_report(evaluation)
        
        # Prepare output
        output = {
            'test_city': self.test_city,
            'train_cities': self.train_cities,
            'total_test_utterances': len(classification_results),
            'evaluation': evaluation,
            'classifications': classification_results
        }
        
        # Save results
        if save_results:
            output_file = f"results_{self.test_city}.json"
            print(f"\nSaving results to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        
        return output


def main():
    """Main function for command-line usage"""
    import sys
    
    # Default cities (modify based on your actual cities)
    default_cities = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF']  # Update with your actual city codes
    
    if len(sys.argv) > 1:
        test_city = sys.argv[1].upper()
    else:
        test_city = input(f"Enter test city code (available: {default_cities}): ").strip().upper()
    
    if test_city not in default_cities:
        print(f"Error: {test_city} not in available cities: {default_cities}")
        return
    
    # Create pipeline
    pipeline = TrainTestPipeline(
        cities=default_cities,
        test_city=test_city
    )
    
    # Run pipeline
    results = pipeline.run(batch_size=20)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"Test City: {test_city}")
    print(f"Train Cities: {', '.join(pipeline.train_cities)}")
    
    if 'combined' in results['evaluation']:
        combined = results['evaluation']['combined']
        print(f"\nCombined Classification (Public Comment OR Hearing):")
        print(f"  Accuracy:  {combined['accuracy']:.4f}")
        print(f"  F1 Score:  {combined['f1']:.4f}")
        print(f"  Precision: {combined['precision']:.4f}")
        print(f"  Recall:    {combined['recall']:.4f}")
        print(f"  Support:   {combined['support']}")


if __name__ == "__main__":
    main()
