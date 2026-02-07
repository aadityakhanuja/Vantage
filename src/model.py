"""
model.py
Train Random Forest credit risk model and generate predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


class CreditRiskModel:
    """
    Credit risk prediction model using behavioral features.
    """
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def train(self, feature_df):
        """
        Train the Random Forest model.
        """
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        # Prepare data
        X = feature_df[self.feature_names]
        y = feature_df['default']
        
        print(f"\nDataset:")
        print(f"  - Total customers: {len(X):,}")
        print(f"  - Features: {len(self.feature_names)}")
        print(f"  - Default rate: {y.mean():.1%}")
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  - Training set: {len(self.X_train):,} customers")
        print(f"  - Test set: {len(self.X_test):,} customers")
        
        # Train Random Forest
        print(f"\nTraining Random Forest...")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print(f"✓ Training complete")
        
        # Evaluate
        self.evaluate()
        
        return self.model
    
    def evaluate(self):
        """
        Evaluate model performance.
        """
        print("\n" + "="*70)
        print("MODEL PERFORMANCE")
        print("="*70)
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, y_pred, 
            target_names=['No Default', 'Default'],
            digits=3
        ))
        
        # AUC
        auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"ROC-AUC Score: {auc:.3f}")
        
        # Feature importance
        self.print_feature_importance()
        
        return {
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def print_feature_importance(self):
        """
        Print feature importance rankings.
        """
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE")
        print("="*70)
        
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.iterrows():
            print(f"  {row['feature']:.<50} {row['importance']:.3f}")
        
        return importances
    
    def predict_customer(self, customer_features):
        """
        Generate prediction and explanation for a single customer.
        """
        # Convert to DataFrame
        X = pd.DataFrame([customer_features])[self.feature_names]
        
        # Predict
        risk_proba = self.model.predict_proba(X)[0, 1]
        risk_pred = self.model.predict(X)[0]
        
        # Determine risk level
        if risk_proba > 0.70:
            risk_level = "HIGH"
        elif risk_proba > 0.40:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Generate explanation
        explanation = self.explain_prediction(customer_features, risk_proba)
        
        return {
            'risk_probability': risk_proba,
            'risk_prediction': risk_pred,
            'risk_level': risk_level,
            'explanation': explanation
        }
    
    def explain_prediction(self, customer_features, risk_proba):
        """
        Generate human-readable explanation of prediction.
        """
        concerns = []
        strengths = []
        
        # Late-night transactions
        if 'late_night_pct' in customer_features:
            if customer_features['late_night_pct'] > 0.15:
                concerns.append(f"High late-night spending ({customer_features['late_night_pct']:.1%})")
            elif customer_features['late_night_pct'] < 0.05:
                strengths.append(f"Low late-night spending ({customer_features['late_night_pct']:.1%})")
        
        # Velocity spikes
        if 'velocity_spike' in customer_features:
            if customer_features['velocity_spike'] > 4.0:
                concerns.append(f"Binge spending episodes (spike: {customer_features['velocity_spike']:.1f}x)")
            elif customer_features['velocity_spike'] < 2.5:
                strengths.append(f"Consistent spending pattern")
        
        # Declined transactions
        if 'declined_rate' in customer_features:
            if customer_features['declined_rate'] > 0.08:
                concerns.append(f"High decline rate ({customer_features['declined_rate']:.1%})")
            elif customer_features['declined_rate'] < 0.03:
                strengths.append(f"Minimal payment issues")
        
        return {
            'risk_probability': risk_proba,
            'concerns': concerns,
            'strengths': strengths
        }
    
    def generate_decision_report(self, customer_id, customer_features, prediction):
        """
        Generate formatted decision support report for loan officer.
        """
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║               CREDIT RISK DECISION SUPPORT REPORT                    ║
╠══════════════════════════════════════════════════════════════════════╣
║ Customer ID: {customer_id:<56} ║
║ Risk Level:  {prediction['risk_level']:<20} ({prediction['risk_probability']:.1%} default probability)  ║
╠══════════════════════════════════════════════════════════════════════╣
║ KEY BEHAVIORAL INDICATORS                                            ║
╠══════════════════════════════════════════════════════════════════════╣
"""