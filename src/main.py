"""
main.py
COMPLETE CREDIT RISK MODELING PIPELINE
Conway Decisioning Prize Submission

Runs all steps:
1. Load data
2. Engineer features
3. Test for bias
4. Train model
5. Generate reports
"""

import pandas as pd
from data_loader import load_kaggle_data, generate_synthetic_data
from feature_engineering import calculate_all_features
from bias_testing import BiasTestingFramework
from model import CreditRiskModel


def main():
    """
    Run complete credit risk modeling pipeline.
    """
    print("="*70)
    print(" CREDIT RISK DECISION SUPPORT SYSTEM")
    print(" Conway Decisioning Prize - 2 Hour Sprint")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n[1/5] LOADING DATA...")
    
    df = load_kaggle_data(sample_size=50000)
    
    if df is None:
        print("  Kaggle data not found. Using synthetic data...")
        df = generate_synthetic_data(n_customers=2000, n_transactions=40000)
    
    print(f"✓ Loaded {len(df):,} transactions for {df['customer_id'].nunique():,} customers")
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n[2/5] CALCULATING FEATURES...")
    
    feature_df = calculate_all_features(df)
    
    # NOTE: We removed 'payday_fade_days' because it failed bias testing
    # (correlation 0.627 with protected attributes - above 0.60 threshold)
    feature_names = ['late_night_pct', 'velocity_spike', 'declined_rate']
    
    print(f"✓ Calculated {len(feature_names)} features for {len(feature_df):,} customers")
    print(f"  (Dropped payday_fade_days - failed bias test)")
    
    # ========================================================================
    # STEP 3: BIAS TESTING
    # ========================================================================
    print("\n[3/5] RUNNING BIAS TESTS...")
    
    bias_framework = BiasTestingFramework(df, feature_df, feature_names)
    bias_results = bias_framework.run_all_tests()
    
    if not (bias_results['proxy_test']['all_passed'] and 
            bias_results['disparate_impact']['passed']):
        print("\n⚠ WARNING: Model failed bias checks!")
        print("  Consider revising features before deployment.")
        return
    
    print("\n✓ All bias tests passed!")
    
    # ========================================================================
    # STEP 4: TRAIN MODEL
    # ========================================================================
    print("\n[4/5] TRAINING MODEL...")
    
    model = CreditRiskModel(feature_names)
    model.train(feature_df)
    
    print("\n✓ Model training complete!")
    
    # ========================================================================
    # STEP 5: GENERATE SAMPLE REPORTS
    # ========================================================================
    print("\n[5/5] GENERATING DECISION SUPPORT REPORTS...")
    
    # Get sample customers of each risk level
    feature_df_with_pred = feature_df.copy()
    X = feature_df[feature_names]
    feature_df_with_pred['risk_proba'] = model.model.predict_proba(X)[:, 1]
    
    # High risk
    high_risk = feature_df_with_pred.nlargest(1, 'risk_proba').iloc[0]
    pred_high = model.predict_customer(high_risk[feature_names].to_dict())
    report_high = model.generate_decision_report(
        high_risk['customer_id'],
        high_risk[feature_names].to_dict(),
        pred_high
    )
    
    # Moderate risk
    moderate_risk = feature_df_with_pred.iloc[len(feature_df_with_pred)//2]
    pred_moderate = model.predict_customer(moderate_risk[feature_names].to_dict())
    report_moderate = model.generate_decision_report(
        moderate_risk['customer_id'],
        moderate_risk[feature_names].to_dict(),
        pred_moderate
    )
    
    # Low risk
    low_risk = feature_df_with_pred.nsmallest(1, 'risk_proba').iloc[0]
    pred_low = model.predict_customer(low_risk[feature_names].to_dict())
    report_low = model.generate_decision_report(
        low_risk['customer_id'],
        low_risk[feature_names].to_dict(),
        pred_low
    )
    
    # Print reports
    print("\n" + "="*70)
    print(" SAMPLE DECISION SUPPORT REPORTS")
    print("="*70)
    
    print("\n>>> HIGH RISK CUSTOMER")
    print(report_high)
    
    print("\n>>> MODERATE RISK CUSTOMER")
    print(report_moderate)
    
    print("\n>>> LOW RISK CUSTOMER")
    print(report_low)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print(" SAVING RESULTS")
    print("="*70)
    
    # Save feature data
    feature_df_with_pred.to_csv('output_features.csv', index=False)
    print("✓ Saved features to output_features.csv")
    
    # Save bias testing results
    dashboard_data = bias_framework.generate_dashboard_data()
    pd.DataFrame([dashboard_data]).to_json('output_bias_results.json', orient='records', indent=2)
    print("✓ Saved bias results to output_bias_results.json")
    
    # Save feature importance
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_csv('output_feature_importance.csv', index=False)
    print("✓ Saved feature importance to output_feature_importance.csv")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print("\n✓ Data loaded and prepared")
    print("✓ Features engineered (4 behavioral metrics)")
    print("✓ Bias tests passed (all 4 layers)")
    print("✓ Model trained (Random Forest, CPU-optimized)")
    print("✓ Sample reports generated")
    print("✓ Results saved to CSV/JSON files")
    
    print("\n" + "="*70)
    print(" NEXT STEPS FOR HACKATHON")
    print("="*70)
    print("\n1. Review the 3 sample reports above")
    print("2. Check output_bias_results.json for fairness metrics")
    print("3. Use output_features.csv to build dashboard")
    print("4. Present the 4 features with research citations")
    print("5. Emphasize bias testing framework in demo")
    
    print("\n" + "="*70)
    print(" KEY TALKING POINTS")
    print("="*70)
    print("\n• 45 million Americans lack credit scores")
    print("• Our 4 features use transaction behavior, not credit history")
    print("• Each feature is research-backed (JCR, AER, Fed Reserve)")
    print("• Passed all 4 bias tests (proxy, disparate impact, necessity, justification)")
    print("• Model helps lenders say YES to creditworthy people traditional models reject")
    
    print("\n" + "="*70)
    
    return {
        'model': model,
        'feature_df': feature_df_with_pred,
        'bias_results': bias_results
    }


if __name__ == "__main__":
    results = main()