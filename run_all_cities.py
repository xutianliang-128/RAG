"""
Runner script to train and test on all cities
Loops through each city, trains on others, and tests on the current city
"""
import os
import json
from typing import List, Dict, Any
from train_test import TrainTestPipeline
from evaluation import print_evaluation_report


class AllCitiesRunner:
    """Runner for all cities"""
    
    def __init__(self, cities: List[str], data_dir: str = ".", batch_size: int = 20):
        """
        Initialize runner
        
        Args:
            cities: List of all city codes
            data_dir: Directory containing data files
            batch_size: Batch size for classification
        """
        self.cities = cities
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.all_results = []
    
    def run_all(self, save_individual: bool = True, save_summary: bool = True) -> Dict[str, Any]:
        """
        Run training and testing for all cities
        
        Args:
            save_individual: Whether to save individual city results
            save_summary: Whether to save summary report
            
        Returns:
            Dictionary with all results and summary
        """
        print("=" * 80)
        print("ALL CITIES TRAINING AND TESTING")
        print("=" * 80)
        print(f"Total cities: {len(self.cities)}")
        print(f"Cities: {', '.join(self.cities)}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 80)
        
        all_results = {}
        summary_metrics = []
        
        for i, test_city in enumerate(self.cities, 1):
            print(f"\n{'='*80}")
            print(f"City {i}/{len(self.cities)}: {test_city}")
            print(f"{'='*80}")
            
            try:
                # Create pipeline for this city
                pipeline = TrainTestPipeline(
                    cities=self.cities,
                    test_city=test_city,
                    data_dir=self.data_dir
                )
                
                # Run training and testing
                results = pipeline.run(
                    batch_size=self.batch_size,
                    save_results=save_individual
                )
                
                # Store results
                all_results[test_city] = results
                
                # Extract metrics for summary
                if 'combined' in results['evaluation']:
                    metrics = results['evaluation']['combined']
                    summary_metrics.append({
                        'test_city': test_city,
                        'train_cities': results['train_cities'],
                        'total_utterances': results['total_test_utterances'],
                        'accuracy': metrics['accuracy'],
                        'f1': metrics['f1'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'support': metrics.get('support', 0)
                    })
                    
                    # Print quick summary
                    print(f"\n{test_city} Results:")
                    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"  F1 Score:  {metrics['f1']:.4f}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall:    {metrics['recall']:.4f}")
                
            except Exception as e:
                print(f"\nERROR processing {test_city}: {e}")
                import traceback
                traceback.print_exc()
                all_results[test_city] = {
                    'error': str(e),
                    'test_city': test_city
                }
        
        # Generate summary
        summary = self._generate_summary(summary_metrics)
        
        # Save summary report
        if save_summary:
            summary_file = "all_cities_summary.json"
            print(f"\n{'='*80}")
            print("Saving summary report...")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary,
                    'individual_results': all_results
                }, f, indent=2, ensure_ascii=False)
            print(f"Summary saved to {summary_file}")
        
        # Print final summary
        self._print_final_summary(summary)
        
        return {
            'summary': summary,
            'individual_results': all_results
        }
    
    def _generate_summary(self, summary_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not summary_metrics:
            return {
                'total_cities': len(self.cities),
                'successful_runs': 0,
                'failed_runs': len(self.cities),
                'total_test_utterances': 0,
                'average_metrics': {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                },
                'per_city_metrics': [],
                'best_city': None,
                'worst_city': None
            }
        
        # Calculate averages
        accuracies = [m['accuracy'] for m in summary_metrics]
        f1_scores = [m['f1'] for m in summary_metrics]
        precisions = [m['precision'] for m in summary_metrics]
        recalls = [m['recall'] for m in summary_metrics]
        total_utterances = sum(m['support'] for m in summary_metrics)
        
        summary = {
            'total_cities': len(self.cities),
            'successful_runs': len(summary_metrics),
            'failed_runs': len(self.cities) - len(summary_metrics),
            'total_test_utterances': total_utterances,
            'average_metrics': {
                'accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
                'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
                'precision': sum(precisions) / len(precisions) if precisions else 0.0,
                'recall': sum(recalls) / len(recalls) if recalls else 0.0
            },
            'per_city_metrics': summary_metrics,
            'best_city': max(summary_metrics, key=lambda x: x['f1']) if summary_metrics else None,
            'worst_city': min(summary_metrics, key=lambda x: x['f1']) if summary_metrics else None
        }
        
        return summary
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary report"""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - ALL CITIES")
        print("=" * 80)
        
        print(f"\nTotal cities processed: {summary['total_cities']}")
        print(f"Successful runs: {summary['successful_runs']}")
        print(f"Failed runs: {summary['failed_runs']}")
        
        if summary['successful_runs'] > 0:
            print(f"Total test utterances: {summary.get('total_test_utterances', 0)}")
        else:
            print("No successful runs - check error messages above")
        
        if summary['successful_runs'] > 0:
            avg = summary['average_metrics']
            print(f"\nAverage Metrics Across All Cities:")
            print(f"  Accuracy:  {avg['accuracy']:.4f}")
            print(f"  F1 Score:  {avg['f1']:.4f}")
            print(f"  Precision: {avg['precision']:.4f}")
            print(f"  Recall:    {avg['recall']:.4f}")
            
            if summary['best_city']:
                best = summary['best_city']
                print(f"\nBest Performance: {best['test_city']}")
                print(f"  F1 Score: {best['f1']:.4f}")
                print(f"  Accuracy: {best['accuracy']:.4f}")
            
            if summary['worst_city']:
                worst = summary['worst_city']
                print(f"\nWorst Performance: {worst['test_city']}")
                print(f"  F1 Score: {worst['f1']:.4f}")
                print(f"  Accuracy: {worst['accuracy']:.4f}")
            
            print(f"\n{'='*80}")
            print("Per-City Results:")
            print(f"{'='*80}")
            print(f"{'City':<10} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12} {'Support':<10}")
            print("-" * 80)
            
            for metrics in summary['per_city_metrics']:
                print(f"{metrics['test_city']:<10} "
                      f"{metrics['accuracy']:<12.4f} "
                      f"{metrics['f1']:<12.4f} "
                      f"{metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} "
                      f"{metrics['support']:<10}")
        
        print("=" * 80)


def discover_cities(data_dir: str = "data") -> List[str]:
    """
    Automatically discover all cities from data directory
    
    Args:
        data_dir: Directory containing train/test files
        
    Returns:
        List of city codes found
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return []
    
    cities = set()
    
    # Look for all train files
    for filename in os.listdir(data_dir):
        if filename.endswith("_train.json"):
            city = filename.replace("_train.json", "")
            cities.add(city)
    
    # Verify test files exist for each city
    valid_cities = []
    for city in sorted(cities):
        test_file = os.path.join(data_dir, f"{city}_test.json")
        if os.path.exists(test_file):
            valid_cities.append(city)
        else:
            print(f"Warning: Found {city}_train.json but no {city}_test.json")
    
    return valid_cities


def main():
    """Main function"""
    import sys
    
    # Data directory
    data_dir = "data"
    
    # Discover cities automatically
    print("=" * 80)
    print("Discovering cities from data directory...")
    print("=" * 80)
    
    cities = discover_cities(data_dir)
    
    if not cities:
        print("Error: No valid city data found!")
        print(f"Please ensure data files are in '{data_dir}/' directory")
        print("Expected format: {CITY}_train.json and {CITY}_test.json")
        return
    
    print(f"\nFound {len(cities)} cities: {', '.join(cities)}")
    
    # Allow override via command line
    if len(sys.argv) > 1:
        specified_cities = [c.upper() for c in sys.argv[1:]]
        # Filter to only include cities that exist
        cities = [c for c in specified_cities if c in cities]
        if not cities:
            print(f"Error: None of the specified cities found in data directory")
            return
        print(f"Using specified cities: {', '.join(cities)}")
    
    # Confirm before running
    print(f"\nWill process {len(cities)} cities:")
    for city in cities:
        print(f"  - {city}: train on {[c for c in cities if c != city]}, test on {city}")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Exiting...")
        return
    
    # Create runner
    runner = AllCitiesRunner(
        cities=cities,
        data_dir=data_dir,
        batch_size=20
    )
    
    # Run all cities
    try:
        results = runner.run_all(
            save_individual=True,
            save_summary=True
        )
        print("\n" + "=" * 80)
        print("ALL CITIES PROCESSING COMPLETED!")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be saved.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
