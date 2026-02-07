"""
feature_engineering.py
Calculate the 4 creative credit risk features
"""

import pandas as pd
import numpy as np


def calculate_all_features(df):
    """
    Calculate all 4 features for each customer.
    
    Features:
    1. Late-Night Transaction % (11PM-4AM)
    2. Transaction Velocity Spikes  
    3. Declined Transaction Rate
    4. Payday Fade (days until 50% spent)
    
    Returns:
        DataFrame with one row per customer and all features
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    customers = []
    
    for customer_id in df['customer_id'].unique():
        customer_data = df[df['customer_id'] == customer_id].copy()
        
        features = {
            'customer_id': customer_id,
            'default': customer_data['default'].iloc[0],  # Target variable
        }
        
        # Calculate each feature
        features.update(calculate_late_night_pct(customer_data))
        features.update(calculate_velocity_spike(customer_data))
        features.update(calculate_declined_rate(customer_data))
        features.update(calculate_payday_fade(customer_data))
        
        customers.append(features)
    
    feature_df = pd.DataFrame(customers)
    
    print("\n✓ Feature engineering complete")
    print(f"  - Customers processed: {len(feature_df):,}")
    print(f"  - Features created: {len(feature_df.columns) - 2}")  # Minus customer_id and default
    
    return feature_df


def calculate_late_night_pct(customer_data):
    """
    FEATURE 1: Late-Night Transaction Percentage
    
    Measures: % of transactions between 11PM-4AM
    Interpretation: Higher = more impulsive spending behavior
    """
    # Count late-night transactions (23:00-04:00)
    late_night = customer_data[
        (customer_data['hour'] >= 23) | (customer_data['hour'] <= 4)
    ]
    
    # Don't count deposits
    transactions_only = customer_data[customer_data['is_deposit'] == 0]
    late_night_only = late_night[late_night['is_deposit'] == 0]
    
    if len(transactions_only) == 0:
        pct = 0.0
    else:
        pct = len(late_night_only) / len(transactions_only)
    
    return {
        'late_night_pct': pct,
        'late_night_count': len(late_night_only),
        'total_transactions': len(transactions_only)
    }


def calculate_velocity_spike(customer_data):
    """
    FEATURE 2: Transaction Velocity Spike
    
    Measures: Max transactions in 24hr / Average daily transactions
    Interpretation: Higher = binge spending episodes
    """
    # Don't count deposits
    transactions_only = customer_data[customer_data['is_deposit'] == 0].copy()
    
    if len(transactions_only) < 2:
        return {
            'velocity_spike': 1.0,
            'max_daily_transactions': 0,
            'avg_daily_transactions': 0
        }
    
    # Count transactions per day
    daily_counts = transactions_only.groupby('date').size()
    
    max_daily = daily_counts.max()
    avg_daily = daily_counts.mean()
    
    if avg_daily == 0:
        spike = 1.0
    else:
        spike = max_daily / avg_daily
    
    return {
        'velocity_spike': spike,
        'max_daily_transactions': int(max_daily),
        'avg_daily_transactions': float(avg_daily)
    }


def calculate_declined_rate(customer_data):
    """
    FEATURE 3: Declined Transaction Rate
    
    Measures: Declined transactions / Total attempted transactions
    Interpretation: Higher = financial stress, hitting limits
    """
    # Count declines
    declined = customer_data[customer_data['is_declined'] == 1]
    total = len(customer_data[customer_data['is_deposit'] == 0])
    
    if total == 0:
        rate = 0.0
    else:
        rate = len(declined) / total
    
    return {
        'declined_rate': rate,
        'declined_count': len(declined),
        'attempted_transactions': total
    }


def calculate_payday_fade(customer_data):
    """
    FEATURE 4: Payday Fade
    
    Measures: Days until 50% of paycheck is spent
    Interpretation: Lower = poor budgeting, no financial buffer
    """
    # Identify paycheck deposits (large deposits or marked as income)
    deposits = customer_data[
        (customer_data['is_deposit'] == 1) | 
        (customer_data['merchant_category'] == 'income')
    ].sort_values('timestamp')
    
    if len(deposits) == 0:
        # No payday detected, return neutral value
        return {
            'payday_fade_days': 15.0,  # Neutral value
            'payday_count': 0,
            'avg_paycheck_amount': 0
        }
    
    fade_days_list = []
    
    for idx, payday in deposits.iterrows():
        payday_date = payday['timestamp']
        payday_amount = payday['amount']
        
        # Get all spending after this payday (until next payday or 30 days)
        next_payday_date = payday_date + pd.Timedelta(days=30)
        
        # Find next payday if exists
        future_deposits = deposits[deposits['timestamp'] > payday_date]
        if len(future_deposits) > 0:
            next_payday_date = future_deposits.iloc[0]['timestamp']
        
        # Get spending between this payday and next
        spending_period = customer_data[
            (customer_data['timestamp'] > payday_date) &
            (customer_data['timestamp'] < next_payday_date) &
            (customer_data['is_deposit'] == 0) &
            (customer_data['amount'] > 0)  # Spending only (negative amounts)
        ].sort_values('timestamp')
        
        if len(spending_period) == 0:
            continue
        
        # Calculate cumulative spending
        spending_period = spending_period.copy()
        spending_period['cumulative_spent'] = spending_period['amount'].cumsum()
        
        # Find when 50% of paycheck is gone
        half_paycheck = payday_amount * 0.5
        spent_half = spending_period[spending_period['cumulative_spent'] >= half_paycheck]
        
        if len(spent_half) > 0:
            days_to_half = (spent_half.iloc[0]['timestamp'] - payday_date).days
            fade_days_list.append(max(days_to_half, 1))  # At least 1 day
    
    if len(fade_days_list) == 0:
        avg_fade = 15.0  # Neutral value
    else:
        avg_fade = np.mean(fade_days_list)
    
    return {
        'payday_fade_days': avg_fade,
        'payday_count': len(deposits),
        'avg_paycheck_amount': deposits['amount'].mean()
    }


def get_feature_descriptions():
    """
    Return descriptions of all features for documentation/dashboard.
    """
    return {
        'late_night_pct': {
            'name': 'Late-Night Transaction %',
            'formula': 'transactions_11pm_4am / total_transactions',
            'interpretation': 'Higher = impulsive spending, poor sleep/stress',
            'research': 'Journal of Consumer Research 2019 - Ego depletion theory',
            'good_range': '0.00 - 0.10',
            'concerning_threshold': '> 0.20'
        },
        'velocity_spike': {
            'name': 'Transaction Velocity Spike',
            'formula': 'max_daily_transactions / avg_daily_transactions',
            'interpretation': 'Higher = binge spending episodes, loss of control',
            'research': 'American Economic Review 2021 - Compensatory consumption',
            'good_range': '1.0 - 3.0',
            'concerning_threshold': '> 5.0'
        },
        'declined_rate': {
            'name': 'Declined Transaction Rate',
            'formula': 'declined_transactions / attempted_transactions',
            'interpretation': 'Higher = already hitting limits, cash flow crisis',
            'research': 'Federal Reserve Bank SF 2020 - Payment failure as early warning',
            'good_range': '0.00 - 0.05',
            'concerning_threshold': '> 0.10'
        },
        'payday_fade_days': {
            'name': 'Payday Fade',
            'formula': 'days_until_50pct_paycheck_spent',
            'interpretation': 'Lower = poor budgeting, no emergency buffer',
            'research': 'Kaplan et al. 2022 - Hand-to-mouth consumers',
            'good_range': '> 15 days',
            'concerning_threshold': '< 5 days'
        }
    }


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")
    
    # Import data loader
    from data_loader import load_kaggle_data, generate_synthetic_data
    
    # Try to load data
    print("\nLoading data...")
    df = load_kaggle_data()
    
    if df is None:
        print("Using synthetic data...")
        df = generate_synthetic_data(n_customers=100, n_transactions=5000)
    
    # Calculate features
    feature_df = calculate_all_features(df)
    
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    print(feature_df.describe())
    
    print("\n" + "="*70)
    print("SAMPLE CUSTOMERS")
    print("="*70)
    print(feature_df.head(10))
    
    print("\n✓ Feature engineering test complete!")