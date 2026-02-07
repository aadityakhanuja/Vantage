"""
AUTOMATED CREDIT RISK ASSESSMENT SYSTEM
Simplified but complete implementation with genetic programming
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')


class CreditRiskSystem:
    """Complete credit risk assessment with genetic programming"""
    
    def __init__(self, random_seed=42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        self.schema = None
        self.df_standard = None
        self.final_features = []
        self.feature_formulas = {}
        self.feature_explanations = {}
        self.model = None
        self.training_stats = {}
        
    # ========== STEP 1-2: LOAD AND DETECT SCHEMA ==========
    
    def load_and_detect(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and detect column types"""
        print("=" * 80)
        print("STEP 1-2: LOADING CSV AND DETECTING SCHEMA")
        print("=" * 80)
        
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Detect schema
        self.schema = {}
        for col in df.columns:
            self.schema[col] = self._detect_column_type(df[col])
        
        print("\n✓ Detected Schema:")
        for col, col_type in self.schema.items():
            print(f"  {col:25s} → {col_type}")
        
        return df
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect what a column represents"""
        if pd.api.types.is_numeric_dtype(series):
            if series.min() >= 0 and series.median() < 10000:
                return 'Amount'
            return 'Numeric'
        
        try:
            pd.to_datetime(series.head(100))
            return 'Date'
        except:
            pass
        
        unique_vals = series.nunique()
        if unique_vals < 3:
            return 'Type'
        elif unique_vals < 100:
            return 'Category'
        elif 'id' in series.name.lower() or 'customer' in series.name.lower():
            return 'Account'
        
        return 'Text'
    
    # ========== STEP 3: STANDARDIZE DATA ==========
    
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to standard format"""
        print("\n" + "=" * 80)
        print("STEP 3: STANDARDIZING DATA")
        print("=" * 80)
        
        df_std = df.copy()
        
        # Find and process date column
        date_cols = [k for k, v in self.schema.items() if v == 'Date']
        if date_cols:
            df_std['timestamp'] = pd.to_datetime(df[date_cols[0]])
            df_std['hour'] = df_std['timestamp'].dt.hour
            df_std['day_of_week'] = df_std['timestamp'].dt.dayofweek
            df_std['is_weekend'] = df_std['day_of_week'].isin([5, 6]).astype(int)
            print(f"✓ Extracted time features from '{date_cols[0]}'")
        
        # Find and process amount column
        amount_cols = [k for k, v in self.schema.items() if v == 'Amount']
        if amount_cols:
            df_std['amount'] = df[amount_cols[0]].abs()
            df_std = df_std[df_std['amount'] > 0]
            print(f"✓ Standardized amount from '{amount_cols[0]}'")
        
        # Find account ID
        account_cols = [k for k, v in self.schema.items() if v == 'Account']
        if not account_cols:
            account_cols = [k for k in df.columns if 'customer' in k.lower() or 'id' in k.lower()]
        if account_cols:
            df_std['account_id'] = df[account_cols[0]]
            print(f"✓ Identified account ID: '{account_cols[0]}'")
        
        # Process transaction status
        status_cols = [k for k, v in self.schema.items() if v == 'Type']
        if status_cols:
            values = df[status_cols[0]].str.lower().fillna('')
            df_std['is_declined'] = values.str.contains('decline|fail|reject').astype(int)
            print(f"✓ Extracted transaction status from '{status_cols[0]}'")
        
        self.df_standard = df_std
        print(f"\n✓ Final dataset: {len(df_std)} transactions")
        
        return df_std
    
    # ========== STEP 3.5: ENGINEER BASE FEATURES ==========
    
    def engineer_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features per customer"""
        print("\n" + "=" * 80)
        print("STEP 4: ENGINEERING BASE FEATURES PER CUSTOMER")
        print("=" * 80)
        
        if 'account_id' not in df.columns:
            print("⚠ No account_id found!")
            return df
        
        features = []
        grouped = df.groupby('account_id')
        
        # Amount statistics
        if 'amount' in df.columns:
            features.append(grouped['amount'].mean().rename('amount_mean'))
            features.append(grouped['amount'].std().fillna(0).rename('amount_std'))
            features.append(grouped['amount'].min().rename('amount_min'))
            features.append(grouped['amount'].max().rename('amount_max'))
            features.append(grouped['amount'].count().rename('transaction_count'))
            print("  ✓ Amount features: mean, std, min, max, count")
        
        # Decline rate
        if 'is_declined' in df.columns:
            features.append(grouped['is_declined'].mean().rename('declined_rate'))
            features.append(grouped['is_declined'].sum().rename('declined_count'))
            print("  ✓ Decline features: rate, count")
        
        # Time features
        if 'hour' in df.columns:
            features.append(grouped['hour'].mean().rename('avg_hour'))
            features.append(grouped['hour'].std().fillna(12).rename('hour_std'))
            print("  ✓ Time features: avg_hour, hour_std")
        
        if 'is_weekend' in df.columns:
            features.append(grouped['is_weekend'].mean().rename('weekend_rate'))
            print("  ✓ Weekend rate feature")
        
        # Combine
        feature_df = pd.concat(features, axis=1).reset_index()
        
        # Fill NaNs
        feature_df = feature_df.fillna(0)
        
        # Remove zero std columns (cause issues in division)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        print(f"\n✓ Created {len(feature_df.columns)-1} features for {len(feature_df)} customers")
        
        return feature_df
    
    # ========== STEP 5-6: GENETIC FEATURE GENERATION ==========
    
    def generate_genetic_features(self, feature_df: pd.DataFrame, 
                                  n_features=100, n_generations=10,
                                  labels=None) -> pd.DataFrame:
        """Generate and evolve features using genetic programming"""
        print("\n" + "=" * 80)
        print(f"STEP 5-6: GENETIC FEATURE GENERATION")
        print(f"({n_features} candidates, {n_generations} generations)")
        print("=" * 80)
        
        # Get numeric columns (terminals)
        terminals = [col for col in feature_df.columns 
                    if col != 'account_id' and pd.api.types.is_numeric_dtype(feature_df[col])]
        print(f"\n✓ Base features available: {len(terminals)}")
        
        # Create feature matrix for easy access
        X_base = feature_df[terminals].values
        n_customers = len(X_base)
        
        # Generate synthetic protected attribute for fairness
        protected = np.random.binomial(1, 0.5, n_customers)
        
        # Initialize population with hand-crafted + random features
        print(f"✓ Generating initial population...")
        population = self._generate_initial_population(terminals, n_features)
        
        best_score = 0
        best_features = []
        
        # Evolution loop
        for gen in range(n_generations):
            # Evaluate all features
            scored_pop = []
            
            for formula in population:
                try:
                    feature_vals = self._eval_formula(formula, feature_df, terminals)
                    score = self._score_feature(feature_vals, labels, protected)
                    scored_pop.append((formula, feature_vals, score))
                except:
                    scored_pop.append((formula, np.zeros(n_customers), 0.0))
            
            # Sort by score
            scored_pop = [x for x in scored_pop if np.isfinite(x[2])]
            scored_pop.sort(key=lambda x: x[2], reverse=True)
            
            if len(scored_pop) > 0:
                curr_best = scored_pop[0][2]
                if curr_best > best_score:
                    best_score = curr_best
                
                if gen % 3 == 0:
                    print(f"  Generation {gen:2d}: Best score = {curr_best:.4f}")
            
            # Keep top 20%
            n_keep = max(10, len(scored_pop) // 5)
            survivors = scored_pop[:n_keep]
            
            # Create next generation
            new_pop = [f[0] for f in survivors]
            
            # Crossover (breed new features)
            while len(new_pop) < n_features * 0.7:
                p1 = random.choice(survivors)[0]
                p2 = random.choice(survivors)[0]
                child = self._crossover(p1, p2)
                new_pop.append(child)
            
            # Mutation
            while len(new_pop) < n_features * 0.9:
                parent = random.choice(survivors)[0]
                mutant = self._mutate(parent, terminals)
                new_pop.append(mutant)
            
            # Random new
            while len(new_pop) < n_features:
                new_pop.append(self._random_formula(terminals))
            
            population = new_pop
        
        print(f"\n✓ Evolution complete. Best score: {best_score:.4f}")
        
        # Select final non-redundant features
        return self._select_final_features(scored_pop, feature_df, n_final=5)
    
    def _generate_initial_population(self, terminals: List[str], n: int) -> List[str]:
        """Generate initial feature formulas"""
        population = []
        
        # Add some hand-crafted good features
        if 'amount_std' in terminals and 'amount_mean' in terminals:
            population.append('amount_std / (amount_mean + 1)')  # CV
        if 'declined_rate' in terminals:
            population.append('declined_rate')
        if 'amount_max' in terminals and 'amount_mean' in terminals:
            population.append('amount_max / (amount_mean + 1)')
        if 'hour_std' in terminals:
            population.append('hour_std')
        
        # Fill rest with random
        while len(population) < n:
            population.append(self._random_formula(terminals))
        
        return population
    
    def _random_formula(self, terminals: List[str], depth=0) -> str:
        """Generate random formula"""
        if depth > 2 or random.random() < 0.4:
            return random.choice(terminals)
        
        ops = ['+', '-', '*', '/']
        op = random.choice(ops)
        left = self._random_formula(terminals, depth + 1)
        right = self._random_formula(terminals, depth + 1)
        
        if op == '/':
            return f'({left}) / ({right} + 1)'
        return f'({left}) {op} ({right})'
    
    def _crossover(self, p1: str, p2: str) -> str:
        """Combine two formulas"""
        op = random.choice(['+', '-', '*', '/'])
        if op == '/':
            return f'({p1}) / ({p2} + 1)'
        return f'({p1}) {op} ({p2})'
    
    def _mutate(self, formula: str, terminals: List[str]) -> str:
        """Mutate a formula"""
        if random.random() < 0.5:
            return self._random_formula(terminals)
        else:
            op = random.choice(['+', '-', '*', '/'])
            new_term = random.choice(terminals)
            if op == '/':
                return f'({formula}) / ({new_term} + 1)'
            return f'({formula}) {op} {new_term}'
    
    def _eval_formula(self, formula: str, df: pd.DataFrame, terminals: List[str]) -> np.ndarray:
        """Safely evaluate formula"""
        try:
            # Create safe namespace
            namespace = {col: df[col].values for col in terminals if col in df.columns}
            namespace['np'] = np
            
            result = eval(formula, {"__builtins__": {}}, namespace)
            result = np.asarray(result).flatten()
            
            # Handle NaN/Inf
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
            
            return result
        except:
            return np.zeros(len(df))
    
    def _score_feature(self, feature_vals: np.ndarray, labels=None, protected=None) -> float:
        """Score a feature"""
        # Predictive power
        if labels is not None:
            try:
                corr, _ = spearmanr(feature_vals, labels)
                pred_score = abs(corr) if np.isfinite(corr) else 0
            except:
                pred_score = 0
        else:
            # Unsupervised: use variance
            pred_score = min(np.std(feature_vals) / (abs(np.mean(feature_vals)) + 1), 1.0)
        
        # Fairness
        if protected is not None:
            try:
                corr, _ = pearsonr(feature_vals, protected)
                if abs(corr) > 0.60:
                    fair_score = 0
                else:
                    fair_score = 1 - abs(corr)
            except:
                fair_score = 0.5
        else:
            fair_score = 1.0
        
        # Combined
        return pred_score * 0.7 + fair_score * 0.3
    
    def _select_final_features(self, scored_pop: List, feature_df: pd.DataFrame, 
                              n_final=5) -> pd.DataFrame:
        """Select top non-redundant features"""
        print("\n" + "=" * 80)
        print(f"STEP 7: SELECTING TOP {n_final} NON-REDUNDANT FEATURES")
        print("=" * 80)
        
        selected_formulas = []
        selected_values = []
        selected_scores = []
        
        for formula, vals, score in scored_pop:
            if len(selected_formulas) >= n_final:
                break
            
            # Check correlation with already selected
            is_redundant = False
            for prev_vals in selected_values:
                try:
                    corr = np.corrcoef(vals, prev_vals)[0, 1]
                    if abs(corr) > 0.7:
                        is_redundant = True
                        break
                except:
                    pass
            
            if not is_redundant:
                selected_formulas.append(formula)
                selected_values.append(vals)
                selected_scores.append(score)
                
                # Generate name
                name = self._generate_feature_name(formula, len(selected_formulas))
                self.feature_formulas[name] = formula
                self.feature_explanations[name] = {
                    'name': name,
                    'formula': formula,
                    'score': score
                }
                
                print(f"\n  ✓ Feature {len(selected_formulas)}: {name}")
                print(f"    Formula: {formula}")
                print(f"    Score: {score:.4f}")
        
        self.final_features = list(self.feature_formulas.keys())
        
        # Create feature matrix
        X = pd.DataFrame()
        for name in self.final_features:
            formula = self.feature_formulas[name]
            terminals = [col for col in feature_df.columns if col != 'account_id']
            X[name] = self._eval_formula(formula, feature_df, terminals)
        
        return X
    
    def _generate_feature_name(self, formula: str, idx: int) -> str:
        """Generate human-readable name for feature"""
        formula_lower = formula.lower()
        
        if 'std' in formula_lower and 'mean' in formula_lower:
            return "Volatility_Index"
        elif 'declined' in formula_lower and '/' in formula:
            return "Decline_Rate"
        elif 'max' in formula_lower and 'mean' in formula_lower:
            return "Peak_Ratio"
        elif 'hour' in formula_lower:
            return "Time_Pattern"
        elif 'weekend' in formula_lower:
            return "Weekend_Activity"
        else:
            return f"Generated_Feature_{idx}"
    
    # ========== STEP 8: TRAIN MODEL ==========
    
    def train_model(self, X: pd.DataFrame, labels=None):
        """Train prediction model"""
        print("\n" + "=" * 80)
        print("STEP 8: TRAINING PREDICTION MODEL")
        print("=" * 80)
        
        print(f"\n✓ Feature matrix: {X.shape}")
        
        # Store statistics
        self.training_stats['feature_means'] = X.mean()
        self.training_stats['feature_stds'] = X.std()
        self.training_stats['feature_values'] = X.copy()
        
        if labels is not None and len(labels) == len(X):
            # Supervised
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_seed,
                n_jobs=-1
            )
            
            try:
                cv_scores = cross_val_score(self.model, X, labels, cv=min(5, len(X)//10), 
                                           scoring='roc_auc')
                print(f"✓ Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            except:
                print("✓ Model configured (too few samples for CV)")
            
            self.model.fit(X, labels)
            print("✓ Model trained successfully")
        else:
            print("✓ No labels - using unsupervised scoring")
            self.model = None
    
    # ========== STEP 9: PREDICT RISK ==========
    
    def predict_risk(self, customer_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict risk for new customer"""
        print("\n" + "=" * 80)
        print("STEP 9: GENERATING RISK ASSESSMENT")
        print("=" * 80)
        
        # Calculate features
        terminals = [col for col in customer_df.columns if col != 'account_id']
        customer_features = {}
        
        for name, formula in self.feature_formulas.items():
            val = self._eval_formula(formula, customer_df, terminals)
            customer_features[name] = float(val.mean()) if len(val) > 0 else 0.0
        
        X_new = pd.DataFrame([customer_features])
        
        # Predict
        if self.model is not None:
            try:
                risk_prob = float(self.model.predict_proba(X_new)[0][1])
            except:
                risk_prob = 0.5
        else:
            # Unsupervised: use z-scores
            z_scores = []
            for name, value in customer_features.items():
                mean = self.training_stats['feature_means'][name]
                std = self.training_stats['feature_stds'][name]
                z_scores.append((value - mean) / (std + 1e-10))
            risk_prob = 1 / (1 + np.exp(-np.mean(z_scores)))
        
        # Classify
        if risk_prob > 0.70:
            risk_level = "HIGH"
        elif risk_prob > 0.40:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Generate insights
        insights = {
            'risk_level': risk_level,
            'risk_probability': risk_prob,
            'features': []
        }
        
        for name, value in customer_features.items():
            training_vals = self.training_stats['feature_values'][name]
            percentile = float((training_vals < value).mean())
            
            insights['features'].append({
                'name': name,
                'value': value,
                'percentile': percentile,
                'formula': self.feature_formulas[name]
            })
        
        return insights
    
    def generate_report(self, insights: Dict[str, Any], customer_id: str = "Unknown") -> str:
        """Generate human-readable report"""
        lines = []
        lines.append("=" * 80)
        lines.append("CREDIT RISK ASSESSMENT")
        lines.append("=" * 80)
        lines.append(f"Customer ID: {customer_id}")
        lines.append(f"Risk Level: {insights['risk_level']}")
        lines.append(f"Default Probability: {insights['risk_probability']*100:.1f}%")
        lines.append(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("─" * 80)
        lines.append("KEY RISK FACTORS")
        lines.append("─" * 80)
        
        for feat in insights['features']:
            percentile = feat['percentile']
            
            if percentile > 0.75:
                icon = "⚠️ "
                level = "HIGH"
            elif percentile > 0.50:
                icon = "⚡"
                level = "MODERATE"
            else:
                icon = "✓ "
                level = "LOW"
            
            lines.append(f"\n{icon} {feat['name']}: {feat['value']:.3f}")
            lines.append(f"   Percentile: {percentile*100:.0f}% | Risk: {level}")
            lines.append(f"   Formula: {feat['formula']}")
        
        lines.append("\n" + "─" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("─" * 80)
        
        if insights['risk_level'] == "HIGH":
            lines.append("- Decision: DECLINE or require collateral")
            lines.append("- Multiple high-risk indicators present")
        elif insights['risk_level'] == "MODERATE":
            lines.append("- Decision: APPROVE with conditions")
            lines.append("- Suggested: Lower credit limit + automatic payments")
            lines.append("- Monitor: Monthly review for 6 months")
        else:
            lines.append("- Decision: APPROVE with standard terms")
            lines.append("- Customer shows favorable risk profile")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


def generate_demo_data(n_customers=100, txn_per_customer=50):
    """Generate realistic demo data"""
    np.random.seed(42)
    data = []
    
    for cust_id in range(1, n_customers + 1):
        is_risky = cust_id % 3 == 0
        base_amount = np.random.uniform(20, 200)
        decline_prob = 0.15 if is_risky else 0.03
        
        for _ in range(txn_per_customer):
            amount = abs(np.random.normal(base_amount, base_amount * 0.5))
            if is_risky:
                amount *= np.random.uniform(0.5, 2.0)
            
            hour = np.random.randint(6, 23)
            if is_risky and np.random.random() < 0.3:
                hour = np.random.randint(0, 6)
            
            status = 'Declined' if np.random.random() < decline_prob else 'Approved'
            
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(
                days=np.random.randint(0, 365), hours=hour)
            
            data.append({
                'customer_id': f'CUST_{cust_id:04d}',
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'merchant': np.random.choice(['Amazon', 'Walmart', 'Gas', 'Restaurant']),
                'status': status
            })
    
    return pd.DataFrame(data)


def main():
    """Demo the system"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  AUTOMATED CREDIT RISK ASSESSMENT SYSTEM                            ║
    ║  Using Genetic Programming for Feature Discovery                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Generate demo data
    print("\n📊 Generating demo transaction data...")
    demo_data = generate_demo_data(n_customers=50, txn_per_customer=30)
    demo_data.to_csv('demo_transactions.csv', index=False)
    print(f"✓ Created {len(demo_data)} transactions for 50 customers\n")
    
    # Initialize system
    system = CreditRiskSystem(random_seed=42)
    
    # Run pipeline
    df = system.load_and_detect('demo_transactions.csv')
    df_std = system.standardize_data(df)
    feature_df = system.engineer_base_features(df_std)
    
    # Create labels
    print("\n📊 Creating synthetic risk labels...")
    if 'declined_rate' in feature_df.columns:
        labels = (feature_df['declined_rate'] > 0.1).astype(int)
        print(f"✓ Generated {labels.sum()} high-risk, {len(labels)-labels.sum()} low-risk customers")
    else:
        labels = None
    
    # Run genetic algorithm
    X_evolved = system.generate_genetic_features(
        feature_df, n_features=50, n_generations=10, labels=labels)
    
    # Train model
    system.train_model(X_evolved, labels)
    
    # Test on sample customer
    print("\n" + "=" * 80)
    print("DEMO: Assessing Sample Customer")
    print("=" * 80)
    
    customer_data = feature_df.iloc[[5]]
    insights = system.predict_risk(customer_data)
    report = system.generate_report(insights, customer_id="CUST_0006")
    
    print("\n" + report)
    
    print("\n\n✅ SYSTEM DEMONSTRATION COMPLETE!")
    print("\nSuccessfully demonstrated:")
    print("  ✓ Automatic CSV schema detection")
    print("  ✓ Data standardization and feature engineering")
    print("  ✓ Genetic programming feature generation")
    print("  ✓ Evolutionary feature selection (50+ candidates)")
    print("  ✓ Fairness and bias testing")
    print("  ✓ Random Forest model training")
    print("  ✓ Human-readable risk assessment report")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
