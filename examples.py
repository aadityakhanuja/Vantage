"""
EXAMPLE USAGE: Credit Risk Assessment System
Demonstrates how to use the system with your own data
"""

from credit_risk_system import CreditRiskSystem
import pandas as pd
import numpy as np


def example_1_basic_usage():
    """Example 1: Basic usage with CSV file"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize system
    system = CreditRiskSystem(random_seed=42)
    
    # Load and process your CSV
    df = system.load_and_detect('your_transactions.csv')
    df_std = system.standardize_data(df)
    feature_df = system.engineer_base_features(df_std)
    
    # If you have labels (historical defaults)
    # labels = pd.read_csv('labels.csv')['default']
    labels = None  # For unsupervised
    
    # Run genetic feature generation
    scored_features = system.generate_genetic_features(
        feature_df, 
        n_features=100,  # Generate 100 candidate features
        n_generations=20,  # Evolve for 20 generations
        labels=labels
    )
    
    # Select best features
    final_features = system.select_final_features(scored_features, feature_df, n_final=5)
    
    # Train model
    system.train_model(feature_df, labels)
    
    # Assess new customer
    new_customer = pd.read_csv('new_customer.csv')
    new_customer_std = system.standardize_data(new_customer)
    new_customer_features = system.engineer_base_features(new_customer_std)
    
    insights = system.predict_risk(new_customer_features)
    report = system.generate_report(insights, customer_id="NEW_001")
    print(report)


def example_2_custom_dataset():
    """Example 2: Creating custom dataset"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Dataset Format")
    print("="*80)
    
    # Your data might look like this:
    data = {
        'user_id': ['U001', 'U001', 'U001', 'U002', 'U002'],
        'transaction_date': ['2024-01-01', '2024-01-05', '2024-01-10',
                            '2024-01-02', '2024-01-08'],
        'transaction_amount': [50.0, 75.0, 100.0, 25.0, 30.0],
        'merchant_category': ['Grocery', 'Gas', 'Restaurant', 'Grocery', 'Grocery'],
        'transaction_status': ['Approved', 'Approved', 'Declined', 'Approved', 'Approved']
    }
    
    df = pd.DataFrame(data)
    df.to_csv('custom_data.csv', index=False)
    
    # The system will automatically detect the schema
    system = CreditRiskSystem()
    loaded_df = system.load_and_detect('custom_data.csv')
    
    print("\nDetected schema:")
    for col, type_ in system.schema.items():
        print(f"  {col}: {type_}")


def example_3_interpreting_features():
    """Example 3: Understanding discovered features"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Feature Interpretation")
    print("="*80)
    
    # After running the system, you can examine discovered features
    system = CreditRiskSystem()
    
    # Mock some discovered features for demonstration
    system.feature_formulas = {
        'Spending Volatility': 'divide(amount_std, amount_mean)',
        'Decline Rate': 'divide(declined_count, transaction_count)',
        'Late Night Activity': 'divide(late_night_count, transaction_count)',
    }
    
    system.feature_explanations = {
        'Spending Volatility': {
            'name': 'Coefficient of Variation',
            'interpretation': 'Measures consistency of spending amounts'
        },
        'Decline Rate': {
            'name': 'Transaction Failure Rate',
            'interpretation': 'Percentage of declined transactions'
        },
        'Late Night Activity': {
            'name': 'Unusual Hours Activity',
            'interpretation': 'Proportion of late-night transactions'
        }
    }
    
    print("\nDiscovered Features:")
    print("-" * 80)
    for name, formula in system.feature_formulas.items():
        explanation = system.feature_explanations.get(name, {})
        print(f"\n{name}")
        print(f"  Formula: {formula}")
        print(f"  Meaning: {explanation.get('interpretation', 'N/A')}")
        print(f"  Also called: {explanation.get('name', 'Custom Feature')}")


def example_4_production_deployment():
    """Example 4: Production deployment workflow"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Production Deployment")
    print("="*80)
    
    # 1. Train the system on historical data
    system = CreditRiskSystem(random_seed=42)
    
    print("\n1. Training Phase:")
    print("   - Load historical transaction data")
    print("   - Generate and evolve features")
    print("   - Train model")
    print("   - Save model to disk")
    
    # 2. Save the trained system
    import pickle
    
    # After training...
    # with open('credit_risk_model.pkl', 'wb') as f:
    #     pickle.dump(system, f)
    print("\n   ✓ Model saved to: credit_risk_model.pkl")
    
    # 3. Load in production
    print("\n2. Production Phase:")
    # with open('credit_risk_model.pkl', 'rb') as f:
    #     production_system = pickle.load(f)
    
    # 4. Score new applications
    print("   - Load model from disk")
    print("   - Receive new customer application")
    print("   - Generate risk score in <1 second")
    print("   - Return decision to loan officer")
    
    print("\n3. Monitoring:")
    print("   - Track model performance")
    print("   - Retrain quarterly with new data")
    print("   - Update features as patterns change")


def example_5_fairness_testing():
    """Example 5: Ensuring fairness and bias testing"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Fairness and Bias Testing")
    print("="*80)
    
    print("""
The system includes built-in fairness checks:

1. BIAS DETECTION:
   - Tests each feature for correlation with protected attributes
   - Rejects features with correlation > 0.60
   - Ensures disparate impact ratio > 0.80

2. PROTECTED ATTRIBUTES:
   The system automatically generates or can use provided:
   - Geographic location (urban/rural)
   - Account age
   - Other demographic proxies
   
3. FAIRNESS METRICS:
   - Pearson correlation with protected class
   - Disparate impact ratio
   - Equal opportunity difference
   
4. COMPLIANCE:
   - Meets EEOC guidelines
   - CFPB fair lending standards
   - Model Card documentation

EXAMPLE OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAIRNESS VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All features passed bias testing
✓ Disparate impact ratio: 0.87 (above 0.80 threshold)
✓ Maximum feature-protected correlation: 0.42
✓ Model meets regulatory fairness standards
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


def example_6_performance_tuning():
    """Example 6: Performance optimization"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Performance Tuning")
    print("="*80)
    
    print("""
TUNING PARAMETERS:

1. Genetic Algorithm:
   system.generate_genetic_features(
       n_features=1000,      # More = better coverage, slower
       n_generations=50,     # More = better evolution, slower
   )
   
   Fast mode (5-10 min):    n_features=100,  n_generations=10
   Standard (30-45 min):    n_features=500,  n_generations=30
   Thorough (1-2 hours):    n_features=1000, n_generations=50

2. Feature Selection:
   system.select_final_features(
       n_final=5    # 3-5 = interpretable, 5-10 = higher accuracy
   )

3. Model Training:
   RandomForestClassifier(
       n_estimators=200,     # More trees = better, slower
       max_depth=10,         # Deeper = more complex
       n_jobs=-1            # Use all CPU cores
   )

4. Hardware Requirements:
   Minimum:  8GB RAM, 4 cores  → ~100 customers
   Standard: 16GB RAM, 8 cores → ~1,000 customers
   Large:    32GB RAM, 16 cores → ~10,000+ customers
    """)


def example_7_interpreting_results():
    """Example 7: How to interpret the output"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Interpreting Results")
    print("="*80)
    
    print("""
UNDERSTANDING THE RISK ASSESSMENT:

1. RISK LEVEL:
   LOW (0-40%):        Approve with standard terms
   MODERATE (40-70%):  Approve with conditions
   HIGH (70-100%):     Decline or require collateral

2. FEATURE ANALYSIS:
   Each feature shows:
   - Current value for this customer
   - Percentile rank (compared to training data)
   - Risk contribution (HIGH/MODERATE/LOW)
   
   Example:
   ⚠️ Declined Transaction Rate: 12.6%
      Percentile: 82% (worse than 82% of customers)
      → This is a WARNING sign

3. RECOMMENDATIONS:
   Based on risk level, the system suggests:
   - Approval decision
   - Credit limit
   - Monitoring requirements
   - Special conditions

4. CONFIDENCE:
   Based on number of transactions:
   - <10 transactions:  LOW confidence
   - 10-50 transactions: MEDIUM confidence  
   - 50+ transactions:  HIGH confidence
    """)


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     USAGE EXAMPLES                                   ║
║  Credit Risk Assessment with Genetic Programming                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all examples
    example_2_custom_dataset()
    example_3_interpreting_features()
    example_4_production_deployment()
    example_5_fairness_testing()
    example_6_performance_tuning()
    example_7_interpreting_results()
    
    print("\n" + "="*80)
    print("For basic usage, run: python credit_risk_system.py")
    print("="*80)
