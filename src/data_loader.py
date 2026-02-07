"""
data_loader.py
Load and prepare transaction data for credit risk modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


def load_kaggle_data(filepath='fraudTrain.csv', sample_size=50000):
    """
    Load Kaggle fraud detection dataset and prepare for our use case.
    
    Args:
        filepath: Path to fraudTrain.csv
        sample_size: Number of transactions to use (for speed)
    
    Returns:
        DataFrame with prepared transaction data
    """
    print(f"Loading Kaggle data from {filepath}...")
    
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} transactions")
        
        # Sample for speed (use first N transactions)
        if len(df) > sample_size:
            df = df.head(sample_size)
            print(f"✓ Using {sample_size:,} transactions for speed")
        
        # Rename columns
        df = df.rename(columns={
            'trans_date_trans_time': 'timestamp',
            'cc_num': 'customer_id',
            'amt': 'amount',
            'trans_num': 'transaction_id',
            'category': 'merchant_category',
            'is_fraud': 'is_fraud'
        })
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Add synthetic fields that Kaggle doesn't have
        df = add_synthetic_fields(df)
        
        # Create default labels (we'll use fraud as proxy for default risk)
        # In reality, fraud and default are different, but for demo purposes
        # we'll create a composite risk score
        df['default'] = create_default_labels(df)
        
        print(f"✓ Data preparation complete")
        print(f"  - Customers: {df['customer_id'].nunique():,}")
        print(f"  - Transactions: {len(df):,}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Could not find {filepath}")
        print("  Please download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        print("  Or use generate_synthetic_data() instead")
        return None


def add_synthetic_fields(df):
    """
    Add fields that Kaggle data doesn't have but we need for our 4 features.
    """
    print("Adding synthetic fields...")
    
    # 1. DECLINED TRANSACTIONS
    # Simulate 5% overall decline rate, higher for large amounts
    base_decline_prob = 0.05
    df['is_declined'] = 0
    
    # Higher amounts more likely to decline
    for idx, row in df.iterrows():
        if row['amount'] > df['amount'].quantile(0.90):
            decline_prob = 0.15  # 15% for high amounts
        else:
            decline_prob = base_decline_prob
        
        df.at[idx, 'is_declined'] = np.random.choice([0, 1], p=[1-decline_prob, decline_prob])
    
    print(f"  - Added declined transactions ({df['is_declined'].sum():,} declines)")
    
    # 2. PAYCHECK DEPOSITS
    # For each customer, add income deposits
    deposits = []
    customers = df['customer_id'].unique()[:2000]  # First 2000 customers for speed
    
    for customer_id in customers:
        customer_txns = df[df['customer_id'] == customer_id]
        
        # Find date range for this customer
        min_date = customer_txns['timestamp'].min()
        max_date = customer_txns['timestamp'].max()
        
        # Add deposits every 15 days (bi-weekly paycheck)
        current_date = min_date
        while current_date <= max_date:
            deposits.append({
                'customer_id': customer_id,
                'transaction_id': f'DEP_{customer_id}_{current_date.date()}',
                'timestamp': current_date.replace(hour=9, minute=0),
                'date': current_date.date(),
                'hour': 9,
                'day_of_week': current_date.dayofweek,
                'amount': np.random.uniform(1500, 4000),  # Paycheck size
                'merchant_category': 'income',
                'is_fraud': 0,
                'is_declined': 0,
                'is_deposit': 1
            })
            current_date += timedelta(days=15)
    
    # Combine deposits with transactions
    deposits_df = pd.DataFrame(deposits)
    df['is_deposit'] = 0
    df = pd.concat([df, deposits_df], ignore_index=True)
    df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
    
    print(f"  - Added income deposits ({len(deposits):,} deposits)")
    
    return df


def create_default_labels(df):
    """
    Create default risk labels based on behavioral patterns.
    
    In real world, you'd have historical default data.
    For demo, we'll create labels based on risky behaviors.
    """
    print("Creating default labels...")
    
    # Calculate risk factors per customer
    customer_risk = df.groupby('customer_id').agg({
        'is_fraud': 'mean',  # Fraud history
        'is_declined': 'mean',  # Decline rate
        'amount': 'std',  # Spending volatility
    }).reset_index()
    
    customer_risk.columns = ['customer_id', 'fraud_rate', 'decline_rate', 'amount_std']
    
    # Normalize to 0-1 scale
    customer_risk['fraud_rate_norm'] = (customer_risk['fraud_rate'] - customer_risk['fraud_rate'].min()) / (customer_risk['fraud_rate'].max() - customer_risk['fraud_rate'].min() + 0.0001)
    customer_risk['decline_rate_norm'] = (customer_risk['decline_rate'] - customer_risk['decline_rate'].min()) / (customer_risk['decline_rate'].max() - customer_risk['decline_rate'].min() + 0.0001)
    customer_risk['amount_std_norm'] = (customer_risk['amount_std'] - customer_risk['amount_std'].min()) / (customer_risk['amount_std'].max() - customer_risk['amount_std'].min() + 0.0001)
    
    # Composite risk score
    customer_risk['risk_score'] = (
        0.4 * customer_risk['fraud_rate_norm'] +
        0.3 * customer_risk['decline_rate_norm'] +
        0.3 * customer_risk['amount_std_norm']
    )
    
    # Label top 30% as "default" risk
    threshold = customer_risk['risk_score'].quantile(0.70)
    customer_risk['default'] = (customer_risk['risk_score'] > threshold).astype(int)
    
    # Merge back to main dataframe
    default_map = dict(zip(customer_risk['customer_id'], customer_risk['default']))
    default_labels = df['customer_id'].map(default_map)
    
    print(f"  - Default rate: {default_labels.mean():.1%}")
    
    return default_labels


def generate_synthetic_data(n_customers=2000, n_transactions=50000):
    """
    Generate fully synthetic transaction data if Kaggle data unavailable.
    """
    print(f"Generating synthetic data for {n_customers} customers...")
    
    np.random.seed(42)
    
    # Generate customer IDs
    customers = [f"CUST_{i:05d}" for i in range(n_customers)]
    
    # Generate transactions
    transactions = []
    
    for _ in range(n_transactions):
        customer_id = np.random.choice(customers)
        
        # Random timestamp in last 6 months
        days_ago = np.random.randint(0, 180)
        hour = np.random.randint(0, 24)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=24-hour)
        
        # Transaction amount (log-normal distribution)
        amount = np.random.lognormal(3.5, 1.2)
        
        # Merchant category
        categories = ['grocery', 'gas_transport', 'shopping', 'food_dining', 
                     'entertainment', 'health_fitness', 'travel', 'bills_utilities']
        category = np.random.choice(categories)
        
        # Declined (5% base rate)
        is_declined = np.random.choice([0, 1], p=[0.95, 0.05])
        
        transactions.append({
            'customer_id': customer_id,
            'transaction_id': f'TXN_{len(transactions):08d}',
            'timestamp': timestamp,
            'date': timestamp.date(),
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'amount': amount,
            'merchant_category': category,
            'is_declined': is_declined,
            'is_fraud': 0,
            'is_deposit': 0
        })
    
    df = pd.DataFrame(transactions)
    
    # Add deposits
    for customer_id in customers:
        customer_dates = df[df['customer_id'] == customer_id]['date'].unique()
        if len(customer_dates) > 15:
            # Add paycheck every 15 days
            for i in range(0, len(customer_dates), 15):
                date = customer_dates[i]
                transactions.append({
                    'customer_id': customer_id,
                    'transaction_id': f'DEP_{customer_id}_{date}',
                    'timestamp': pd.Timestamp(date) + pd.Timedelta(hours=9),
                    'date': date,
                    'hour': 9,
                    'day_of_week': pd.Timestamp(date).weekday(),
                    'amount': np.random.uniform(2000, 4500),
                    'merchant_category': 'income',
                    'is_declined': 0,
                    'is_fraud': 0,
                    'is_deposit': 1
                })
    
    df = pd.DataFrame(transactions)
    df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
    
    # Create default labels
    df['default'] = create_default_labels(df)
    
    print(f"✓ Generated {len(df):,} transactions for {n_customers:,} customers")
    
    return df


if __name__ == "__main__":
    # Test the loader
    print("Testing data loader...")
    print("\nAttempting to load Kaggle data...")
    
    df = load_kaggle_data()
    
    if df is None:
        print("\nFalling back to synthetic data...")
        df = generate_synthetic_data()
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    print("\n✓ Data loader test complete!")