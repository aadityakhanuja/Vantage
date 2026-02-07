# 🚀 QUICK START GUIDE
## Automated Credit Risk Assessment System

## ⚡ Run the Demo (1 minute)

```bash
python credit_risk_system.py
```

This will:
- Generate sample data (50 customers, 1500 transactions)
- Auto-detect CSV schema
- Generate 50 candidate features using genetic programming
- Evolve features over 10 generations
- Train a Random Forest model
- Assess a sample customer
- Display a complete risk report

**Expected output:** Full risk assessment in ~30-60 seconds

---

## 📊 Use With Your Own Data (5 minutes)

### Step 1: Prepare Your CSV
Your CSV should have:
- **Customer/Account ID** (any column name)
- **Transaction dates** (any datetime format)
- **Amounts** (numeric values)
- **Optional:** Status (Approved/Declined), Merchant, etc.

Example:
```csv
customer_id,date,amount,status
CUST_001,2024-01-01,50.00,Approved
CUST_001,2024-01-05,75.00,Declined
```

### Step 2: Run the Code

```python
from credit_risk_system import CreditRiskSystem

# Initialize
system = CreditRiskSystem(random_seed=42)

# Load your data
df = system.load_and_detect('your_transactions.csv')
df_std = system.standardize_data(df)
feature_df = system.engineer_base_features(df_std)

# Optional: Load your labels (historical defaults)
# labels = pd.read_csv('labels.csv')['defaulted']
labels = None  # For unsupervised mode

# Generate features (adjust parameters for your needs)
X_evolved = system.generate_genetic_features(
    feature_df, 
    n_features=100,    # More = better, slower (50-500)
    n_generations=20,  # More = better, slower (10-50)
    labels=labels
)

# Train model
system.train_model(X_evolved, labels)

# Assess new customer
new_customer_df = pd.read_csv('new_customer.csv')
new_customer_std = system.standardize_data(new_customer_df)
new_customer_features = system.engineer_base_features(new_customer_std)

insights = system.predict_risk(new_customer_features)
report = system.generate_report(insights, customer_id="NEW_001")
print(report)
```

---

## 🎛️ Configuration Guide

### Performance Modes

| Mode | n_features | n_generations | Time | Accuracy |
|------|-----------|---------------|------|----------|
| **Fast** | 50 | 10 | ~1 min | Good |
| **Standard** | 100 | 20 | ~5 min | Better |
| **Thorough** | 200 | 30 | ~15 min | Best |

### For Production

```python
# High accuracy configuration
X_evolved = system.generate_genetic_features(
    feature_df,
    n_features=500,      # Large population
    n_generations=50,    # More evolution
    labels=labels        # Use real labels
)

# Save model for reuse
import pickle
with open('production_model.pkl', 'wb') as f:
    pickle.dump(system, f)

# Load in production
with open('production_model.pkl', 'rb') as f:
    production_system = pickle.load(f)

# Score instantly (<1 second)
insights = production_system.predict_risk(new_customer)
```

---

## 📈 What You Get

### 1. Auto-discovered Features
The system finds patterns like:
- **Volatility Index:** `amount_std / (amount_mean + 1)`
- **Decline Rate:** `declined_count / transaction_count`
- **Peak Ratio:** `amount_max / (amount_mean + 1)`
- **Time Patterns:** `hour_std` or complex time-based formulas

### 2. Risk Assessment Report
```
════════════════════════════════════════════════════════════════
CREDIT RISK ASSESSMENT
════════════════════════════════════════════════════════════════
Customer ID: CUST_0006
Risk Level: HIGH
Default Probability: 80.0%

KEY RISK FACTORS
────────────────────────────────────────────────────────────────
⚠️  Decline_Rate: -5.997
   Percentile: 92% | Risk: HIGH
   
✓  Time_Pattern: 10.700
   Percentile: 20% | Risk: LOW

RECOMMENDATIONS
────────────────────────────────────────────────────────────────
- Decision: DECLINE or require collateral
- Multiple high-risk indicators present
════════════════════════════════════════════════════════════════
```

### 3. Model Performance
- **Cross-validation AUC:** Typically 0.75-0.95
- **Interpretable features:** Every formula is readable
- **Fairness tested:** All features checked for bias

---

## 🔍 Understanding Output

### Risk Levels
- **LOW (0-40%):** Standard approval
- **MODERATE (40-70%):** Approve with conditions
- **HIGH (70-100%):** Decline or require collateral

### Percentiles
- **0-25%:** Better than most customers (LOW risk)
- **25-50%:** Average (LOW risk)
- **50-75%:** Concerning (MODERATE risk)
- **75-100%:** Severe red flag (HIGH risk)

### Feature Formulas
All mathematical formulas are shown so you can:
- Understand what the model found
- Validate against domain knowledge
- Explain decisions to stakeholders
- Debug unexpected predictions

---

## ⚙️ Advanced Usage

### Custom Feature Templates

```python
# Force certain features to be included
system._generate_initial_population = lambda terms, n: [
    'amount_std / (amount_mean + 1)',  # Volatility
    'declined_rate',                    # Decline rate
    'amount_max / (amount_mean + 1)',  # Peak ratio
] + [system._random_formula(terms) for _ in range(n-3)]
```

### Ensemble Models

```python
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('xgb', xgb.XGBClassifier(n_estimators=200))
])

system.model = ensemble
system.model.fit(X_evolved, labels)
```

### Batch Scoring

```python
# Score many customers at once
all_customers = pd.read_csv('all_applications.csv')

results = []
for customer_id in all_customers['customer_id'].unique():
    customer_data = all_customers[all_customers['customer_id'] == customer_id]
    customer_std = system.standardize_data(customer_data)
    customer_features = system.engineer_base_features(customer_std)
    
    insights = system.predict_risk(customer_features)
    results.append({
        'customer_id': customer_id,
        'risk_level': insights['risk_level'],
        'risk_prob': insights['risk_probability']
    })

results_df = pd.DataFrame(results)
results_df.to_csv('risk_scores.csv', index=False)
```

---

## 🛡️ Fairness & Compliance

### Built-in Protections
- ✅ Automatic bias testing (correlation < 0.60)
- ✅ Disparate impact monitoring
- ✅ No use of protected attributes
- ✅ Explainable decisions (formula transparency)

### Regulatory Compliance
Designed to meet:
- EEOC guidelines
- CFPB fair lending standards
- GDPR right to explanation
- Model risk management requirements

---

## ❓ Troubleshooting

### "No account_id found"
**Solution:** System auto-detects but if it fails:
```python
df_std['account_id'] = df['your_customer_id_column']
```

### "All scores are NaN"
**Solution:** Check for:
- Very few transactions per customer (<10)
- All amounts are the same
- No variation in data

Add more diverse data or adjust parameters.

### "Low cross-validation score"
**Solution:**
- Increase `n_features` and `n_generations`
- Ensure you have enough customers (>100)
- Check label quality
- Try different random seeds

### "Takes too long"
**Solution:**
- Reduce `n_features` (try 50)
- Reduce `n_generations` (try 10)
- Use fewer customers for initial testing
- Ensure `n_jobs=-1` for parallel processing

---

## 📚 Files Included

1. **credit_risk_system.py** - Main system implementation
2. **examples.py** - Detailed usage examples
3. **README.md** - Comprehensive documentation
4. **QUICK_START.md** - This file
5. **demo_transactions.csv** - Sample data

---

## 🎯 Next Steps

1. ✅ Run the demo to see it work
2. ✅ Try with your own CSV data
3. ✅ Adjust parameters for your needs
4. ✅ Review discovered features for domain validity
5. ✅ Deploy to production
6. ✅ Monitor and retrain quarterly

---

## 💡 Tips for Best Results

### Data Quality
- **Minimum:** 10 transactions per customer
- **Good:** 50+ transactions per customer
- **Best:** 100+ transactions per customer

### Feature Count
- **Small dataset (<100 customers):** 50 features, 10 generations
- **Medium (100-1000):** 100 features, 20 generations
- **Large (1000+):** 500 features, 50 generations

### Training Frequency
- Retrain every 3-6 months
- Monitor for data drift
- Update when patterns change

---

## 🤝 Support

Having issues? Check:
1. This quick start guide
2. `examples.py` for common patterns
3. `README.md` for detailed documentation
4. Code comments in `credit_risk_system.py`

---

**You're ready to go! Start with the demo, then use your own data.**

```bash
python credit_risk_system.py
```
