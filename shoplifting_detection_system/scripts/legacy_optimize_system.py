#!/usr/bin/env python3
"""
System Optimization Script
Applies automatic configuration optimization based on current performance
"""

import json
from config_optimizer import ConfigurationOptimizer
from detection.shoplifting_detector import ShopliftingDetector

def main():
    print("🔧 Shoplifting Detection System Optimizer")
    print("=" * 50)
    
    # Load current performance metrics
    try:
        with open('performance_report.json', 'r') as f:
            performance_data = json.load(f)
        current_metrics = performance_data['current_metrics']
        print("📊 Current Performance Metrics:")
        print(f"   Accuracy: {current_metrics['accuracy']:.2f}")
        print(f"   Precision: {current_metrics['precision']:.2f}")
        print(f"   Recall: {current_metrics['recall']:.2f}")
        print(f"   False Positive Rate: {current_metrics['false_positive_rate']:.2f}")
        print(f"   False Negative Rate: {current_metrics['false_negative_rate']:.2f}")
    except FileNotFoundError:
        print("⚠️  No performance report found. Using default metrics.")
        current_metrics = {
            'accuracy': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 1.0
        }
    
    # Initialize optimizer
    optimizer = ConfigurationOptimizer()
    
    print("\n🎯 Current Configuration:")
    for key, value in optimizer.current_config.items():
        print(f"   {key}: {value}")
    
    # Optimize configuration
    print("\n🚀 Optimizing configuration...")
    optimization_result = optimizer.optimize_thresholds(current_metrics)
    
    print("\n✨ Optimization Results:")
    print("Suggestions:")
    for suggestion in optimization_result['suggestions']:
        print(f"   • {suggestion}")
    
    print("\n🔮 Expected Improvements:")
    for metric, prediction in optimization_result['improvements_expected'].items():
        print(f"   {metric}: {prediction}")
    
    print("\n⚙️  New Configuration:")
    for key, value in optimization_result['optimized_config'].items():
        print(f"   {key}: {value}")
    
    # Apply configuration to detector
    print("\n🔄 Applying optimized configuration...")
    detector = ShopliftingDetector()
    success = optimizer.apply_configuration(detector)
    
    if success:
        print("✅ Configuration applied successfully!")
        
        # Verify new thresholds
        print("\n🔍 Verified New Thresholds:")
        print(f"   Shelf Interaction: {detector.shelf_interaction_threshold}")
        print(f"   Concealment: {detector.concealment_threshold}")
        print(f"   Shoplifting: {detector.shoplifting_threshold}")
    else:
        print("❌ Failed to apply configuration!")
    
    # Generate optimization report
    print("\n📋 Generating optimization report...")
    report = optimizer.generate_optimization_report()
    
    with open('optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Optimization report saved to: optimization_report.json")
    
    # Save optimization history
    optimizer.save_optimization_history()
    
    # Show environment-specific recommendations
    print("\n🏪 Environment-Specific Recommendations:")
    environments = ['high_traffic_store', 'small_store', 'jewelry_store', 'grocery_store']
    
    for env in environments:
        config = optimizer.get_recommended_config_for_environment(env)
        print(f"\n   {env.replace('_', ' ').title()}:")
        print(f"      {config.get('description', 'Standard configuration')}")
        print(f"      Shelf Interaction: {config.get('shelf_interaction_threshold', 'N/A')}")
        print(f"      Concealment: {config.get('concealment_threshold', 'N/A')}")
        print(f"      Shoplifting: {config.get('shoplifting_threshold', 'N/A')}")
    
    print("\n🎯 Next Steps:")
    print("1. Test the optimized configuration with realistic scenarios")
    print("2. Monitor performance over the next 24 hours")
    print("3. Run 'python test_realistic_scenarios.py' to validate improvements")
    print("4. Adjust thresholds further if needed based on real-world performance")
    print("5. Consider implementing additional improvements from the report")
    
    print("\n✨ Optimization completed!")

if __name__ == "__main__":
    main()
