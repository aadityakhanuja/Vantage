import pandas as pd
from credit_risk_system import CreditRiskSystem, generate_demo_data

def main():
    print("\n🚦 RUNNING CREDIT RISK TEST ON 5 CLIENTS 🚦\n")

    # -----------------------------
    # Step 1: Train the system
    # -----------------------------
    print("📊 Training model on demo data...")
    demo_data = generate_demo_data(n_customers=40, txn_per_customer=25)
    demo_data.to_csv("demo_train_data.csv", index=False)

    system = CreditRiskSystem(random_seed=42)

    df = system.load_and_detect("demo_train_data.csv")
    df_std = system.standardize_data(df)
    feature_df = system.engineer_base_features(df_std)

    # Create labels
    labels = (feature_df['declined_rate'] > 0.1).astype(int)

    X = system.generate_genetic_features(
        feature_df,
        n_features=30,
        n_generations=6,
        labels=labels
    )

    system.train_model(X, labels)

    # -----------------------------
    # Step 2: Load test clients
    # -----------------------------
    print("\n📂 Loading test clients...")
    test_df = system.load_and_detect("test_clients_5.csv")
    test_df = system.standardize_data(test_df)
    test_features = system.engineer_base_features(test_df)

    # -----------------------------
    # Step 3: Assess risk
    # -----------------------------
    print("\n🔍 RISK RESULTS")
    print("=" * 60)

    high_risk_clients = []

    for _, row in test_features.iterrows():
        customer_id = row['account_id']
        customer_data = pd.DataFrame([row])

        insights = system.predict_risk(customer_data)

        print(f"\nCustomer: {customer_id}")
        print(f"Risk Level: {insights['risk_level']}")
        print(f"Risk Probability: {insights['risk_probability']:.2f}")

        if insights['risk_level'] == "HIGH":
            high_risk_clients.append(customer_id)

    # -----------------------------
    # Final summary
    # -----------------------------
    print("\n" + "=" * 60)
    print("⚠️ HIGH RISK CLIENTS ⚠️")

    if high_risk_clients:
        for cid in high_risk_clients:
            print(f" - {cid}")
    else:
        print("None")

    print("\n✅ TEST COMPLETE\n")

if __name__ == "__main__":
    main()
