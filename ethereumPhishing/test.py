"""
test.py - Synthetic Data Testing (NO DATA LEAKAGE)
Generates synthetic transactions and tests with proper train/test separation
"""

import pandas as pd
import numpy as np
import random
import hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

print("\n" + "=" * 70)
print("  SYNTHETIC DATA TEST (NO DATA LEAKAGE)")
print("=" * 70)

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticEthereumGenerator:
    """Generate realistic synthetic Ethereum transactions"""
    
    def __init__(self):
        self.normal_addresses = []
        self.phishing_addresses = []
        
    def generate_eth_address(self):
        """Generate realistic Ethereum address"""
        random_hex = ''.join(random.choices('0123456789abcdef', k=40))
        return f"0x{random_hex}"
    
    def generate_similar_address(self, original):
        """Generate address poisoning attack target"""
        first = original[:8]
        last = original[-4:]
        middle = ''.join(random.choices('0123456789abcdef', k=32))
        return f"{first}{middle}{last}"
    
    def generate_tx_hash(self):
        """Generate transaction hash"""
        return '0x' + hashlib.sha256(random.randbytes(32)).hexdigest()
    
    def generate_normal_transactions(self, n=800):
        """Generate NORMAL transactions"""
        transactions = []
        num_users = 50
        normal_users = [self.generate_eth_address() for _ in range(num_users)]
        self.normal_addresses = normal_users
        
        contacts = {
            user: [self.generate_eth_address() for _ in range(random.randint(3, 8))]
            for user in normal_users
        }
        
        base_time = int(datetime(2024, 1, 1).timestamp())
        
        for i in range(n):
            from_addr = random.choice(normal_users)
            to_addr = random.choice(contacts[from_addr])
            value = round(random.uniform(0.01, 10.0), 6)
            timestamp = base_time + random.randint(0, 365 * 24 * 3600)
            
            has_contract = random.random() < 0.3
            contract_addr = self.generate_eth_address() if has_contract else None
            input_data = '0x' + ''.join(random.choices('0123456789abcdef', k=random.randint(8, 200))) if has_contract else '0x'
            
            transactions.append({
                'TxHash': self.generate_tx_hash(),
                'BlockHeight': 18000000 + i * random.randint(1, 5),
                'TimeStamp': timestamp,
                'From': from_addr,
                'To': to_addr,
                'Value': value,
                'ContractAddress': contract_addr,
                'Input': input_data,
                'Class': 0
            })
        
        return transactions
    
    def generate_fraud_transactions(self, n=400):
        """Generate FRAUD transactions (dust attacks)"""
        transactions = []
        num_attackers = 20
        attackers = [self.generate_eth_address() for _ in range(num_attackers)]
        self.phishing_addresses = attackers
        
        num_victims = 30
        victims = [self.generate_eth_address() for _ in range(num_victims)]
        
        base_time = int(datetime(2024, 1, 1).timestamp())
        
        for i in range(n):
            attacker = random.choice(attackers)
            victim = random.choice(victims)
            poison_address = self.generate_similar_address(victim)
            
            # KEY FRAUD PATTERNS
            value = 0.0 if random.random() < 0.7 else round(random.uniform(0.0001, 0.0009), 6)
            timestamp = base_time + random.randint(0, 365 * 24 * 3600)
            timestamp += random.randint(0, 300)  # Burst pattern
            
            to_addr = poison_address if random.random() < 0.7 else self.generate_eth_address()
            
            transactions.append({
                'TxHash': self.generate_tx_hash(),
                'BlockHeight': 18000000 + i * random.randint(1, 3),
                'TimeStamp': timestamp,
                'From': attacker,
                'To': to_addr,
                'Value': value,
                'ContractAddress': None,
                'Input': '0x',
                'Class': 1
            })
        
        return transactions
    
    def generate_complete_dataset(self, n_normal=800, n_fraud=400):
        """Generate complete synthetic dataset"""
        print("\n📊 Generating Synthetic Dataset...")
        
        normal_txs = self.generate_normal_transactions(n_normal)
        fraud_txs = self.generate_fraud_transactions(n_fraud)
        
        all_txs = normal_txs + fraud_txs
        df = pd.DataFrame(all_txs)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.sort_values('TimeStamp').reset_index(drop=True)
        
        print(f"   • Total: {len(df):,} transactions")
        print(f"   • Normal: {(df['Class']==0).sum():,} ({(df['Class']==0).sum()/len(df)*100:.1f}%)")
        print(f"   • Fraud:  {(df['Class']==1).sum():,} ({(df['Class']==1).sum()/len(df)*100:.1f}%)")
        
        return df

# ============================================================================
# FEATURE ENGINEERING - NO LEAKAGE
# ============================================================================

class FeatureEngineer:
    """Feature engineering with proper separation"""
    
    def __init__(self):
        self.to_address_seen = defaultdict(set)
        self.address_freq_from = {}
        self.address_freq_to = {}
        self.from_fraud_rates = {}
        self.to_fraud_rates = {}
    
    def fit_statistics(self, df):
        """Fit statistics on training data"""
        self.address_freq_from = df['From'].value_counts().to_dict()
        self.address_freq_to = df['To'].value_counts().to_dict()
        self.from_fraud_rates = df.groupby('From')['Class'].mean().to_dict()
        self.to_fraud_rates = df.groupby('To')['Class'].mean().to_dict()
        return self
    
    def engineer_features(self, df, fit=True):
        """Engineer features without leakage"""
        df = df.copy()
        df['To'] = df['To'].fillna('CONTRACT_CREATION')
        df['Input'] = df['Input'].fillna('0x')
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], unit='s')
        df = df.sort_values('TimeStamp').reset_index(drop=True)
        
        # Dust detection
        df['is_dust'] = (df['Value'] < 0.001).astype(int)
        df['value_log'] = np.log1p(df['Value'])
        df['is_zero_value'] = (df['Value'] == 0).astype(int)
        
        # Address frequency
        if fit:
            self.fit_statistics(df)
        
        df['from_address_freq'] = df['From'].map(self.address_freq_from).fillna(1)
        df['to_address_freq'] = df['To'].map(self.address_freq_to).fillna(1)
        
        # Time features
        df['time_delta_seconds'] = df.groupby('From')['TimeStamp'].diff().dt.total_seconds().fillna(0)
        df['hour_of_day'] = df['TimeStamp'].dt.hour
        df['day_of_week'] = df['TimeStamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # New recipient (reset for each dataset)
        if fit:
            self.to_address_seen = defaultdict(set)
        
        df['is_new_recipient'] = 0
        for idx, row in df.iterrows():
            if row['To'] not in self.to_address_seen[row['From']]:
                df.at[idx, 'is_new_recipient'] = 1
                self.to_address_seen[row['From']].add(row['To'])
        
        # Transaction burst
        df['tx_sequence'] = df.groupby('From').cumcount() + 1
        
        def calculate_burst(group):
            group = group.sort_values('TimeStamp').set_index('TimeStamp')
            group['_temp'] = 1
            group['tx_count_1h'] = group['_temp'].rolling('1H').sum()
            group = group.drop('_temp', axis=1)
            return group.reset_index()
        
        df = df.groupby('From', group_keys=False).apply(calculate_burst)
        df['tx_count_1h'] = df['tx_count_1h'].fillna(1)
        
        # Contract features
        df['has_contract_address'] = (df['ContractAddress'].notna()).astype(int)
        df['has_input_data'] = (df['Input'] != '0x').astype(int)
        df['input_length'] = df['Input'].str.len()
        
        # Target encoding - use training statistics
        df['from_fraud_rate'] = df['From'].map(self.from_fraud_rates).fillna(0.5)
        df['to_fraud_rate'] = df['To'].map(self.to_fraud_rates).fillna(0.5)
        
        # Statistical features
        df['from_avg_value'] = df.groupby('From')['Value'].transform('mean')
        df['value_deviation'] = abs(df['Value'] - df['from_avg_value'])
        
        return df

# ============================================================================
# MODEL TESTING - NO LEAKAGE
# ============================================================================

def test_model_no_leakage(synthetic_df):
    """Test model with proper train/test separation"""
    
    print("\n" + "=" * 70)
    print("  TRAINING & TESTING (PROPER SEPARATION)")
    print("=" * 70)
    
    # CRITICAL: Split BEFORE feature engineering
    print("\n🔀 Splitting Data (70/30) BEFORE Feature Engineering...")
    train_df, test_df = train_test_split(
        synthetic_df, test_size=0.3, random_state=RANDOM_STATE,
        stratify=synthetic_df['Class']
    )
    
    print(f"   • Train: {len(train_df):,} samples")
    print(f"   • Test:  {len(test_df):,} samples")
    
    # Engineer features separately
    print("\n🔧 Engineering Features Separately...")
    
    # Train features - fit statistics
    train_engineer = FeatureEngineer()
    train_engineered = train_engineer.engineer_features(
        train_df.reset_index(drop=True),
        fit=True
    )
    
    # Test features - use ONLY training statistics
    test_engineer = FeatureEngineer()
    test_engineer.address_freq_from = train_engineer.address_freq_from
    test_engineer.address_freq_to = train_engineer.address_freq_to
    test_engineer.from_fraud_rates = train_engineer.from_fraud_rates
    test_engineer.to_fraud_rates = train_engineer.to_fraud_rates
    
    test_engineered = test_engineer.engineer_features(
        test_df.reset_index(drop=True),
        fit=False  # Don't fit on test data
    )
    
    print(f"   • Created {train_engineered.shape[1] - 9} features")
    
    # Prepare features
    print("\n📊 Preparing Features...")
    exclude_cols = ['TxHash', 'BlockHeight', 'TimeStamp', 'From', 'To',
                   'ContractAddress', 'Input', 'Class']
    feature_cols = [col for col in train_engineered.columns if col not in exclude_cols]
    
    X_train = train_engineered[feature_cols].fillna(0)
    y_train = train_engineered['Class']
    X_test = test_engineered[feature_cols].fillna(0)
    y_test = test_engineered['Class']
    
    # Scale (fit on train only)
    print("\n⚖️  Scaling Features (Fit on Train Only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE on training only
    print("\n⚖️  Applying SMOTE (Training Only)...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"   • Balanced: {len(y_train_balanced):,} samples")
    
    # Train model
    print("\n🤖 Training XGBoost Model...")
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=RANDOM_STATE,
        tree_method='hist'
    )
    model.fit(X_train_balanced, y_train_balanced, verbose=False)
    print("   • Training complete!")
    
    # Evaluate on test set
    print("\n📈 Evaluating on Unseen Test Set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results
    print("\n" + "=" * 70)
    print("       FINAL PERFORMANCE (NO DATA LEAKAGE)")
    print("=" * 70)
    print(f"  Precision-Recall AUC:  {pr_auc:.4f}  ⭐")
    print(f"  F1-Score:              {f1:.4f}")
    print(f"  Precision:             {precision:.4f}")
    print(f"  Recall:                {recall:.4f}")
    print(f"  ROC-AUC:               {roc_auc:.4f}")
    print("=" * 70)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0,0]:4d}  |  False Positives: {cm[0,1]:4d}")
    print(f"    False Negatives: {cm[1,0]:4d}  |  True Positives:  {cm[1,1]:4d}")
    print("=" * 70)
    
    # Feature importance
    print("\n🔍 Top 5 Most Important Features:")
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    for i, row in feature_importance.head(5).iterrows():
        print(f"   {i+1}. {row['Feature']:30s} {row['Importance']:.4f}")
    
    # Real-time detection on test set
    print("\n" + "=" * 70)
    print("  REAL-TIME DETECTION TEST (Test Set Only)")
    print("=" * 70)
    
    # Sample from test set
    normal_sample = test_engineered[test_engineered['Class'] == 0].sample(min(2, len(test_engineered[test_engineered['Class'] == 0])))
    fraud_sample = test_engineered[test_engineered['Class'] == 1].sample(min(2, len(test_engineered[test_engineered['Class'] == 1])))
    samples = pd.concat([normal_sample, fraud_sample]).sample(frac=1)
    
    correct = 0
    total = 0
    
    for _, tx in samples.iterrows():
        tx_features = [tx[col] if col in tx.index else 0 for col in feature_cols]
        tx_scaled = scaler.transform([tx_features])
        prob = model.predict_proba(tx_scaled)[0, 1]
        
        actual = "FRAUD" if tx['Class'] == 1 else "NORMAL"
        predicted = "FRAUD" if prob >= 0.5 else "NORMAL"
        status = "✅" if actual == predicted else "❌"
        
        if actual == predicted:
            correct += 1
        total += 1
        
        print(f"\n{status} TX: {tx['TxHash'][:16]}...")
        print(f"   Actual: {actual:6s} | Predicted: {predicted:6s} | Confidence: {prob:6.2%}")
        print(f"   Value: {tx['Value']:.6f} ETH")
        
        if prob >= 0.7 and tx['Value'] < 0.001:
            print(f"   🚨 DUST TRANSACTION - High fraud risk!")
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print(f"  ✅ Sample accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  📊 PR-AUC: {pr_auc:.4f}")
    print(f"  🎯 True Positive Rate: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
    print(f"  ⚠️  False Positive Rate: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}%")
    
    # Performance interpretation
    if pr_auc > 0.95:
        print("\n  ⚠️  Very high performance detected!")
        print("      Possible reasons:")
        print("      1. Synthetic data has very clear patterns")
        print("      2. Fraud signals are strong (dust amounts, bursts)")
        print("      3. This is actually a well-performing model")
    elif pr_auc > 0.85:
        print("\n  ✅ GOOD - Model generalizes well!")
    elif pr_auc > 0.70:
        print("\n  ⚠️  MODERATE - Needs improvement")
    else:
        print("\n  ❌ POOR - Model not learning effectively")
    
    print("=" * 70)
    
    return model, scaler

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete synthetic testing pipeline"""
    
    # Generate synthetic data
    generator = SyntheticEthereumGenerator()
    synthetic_df = generator.generate_complete_dataset(
        n_normal=800,
        n_fraud=400
    )
    
    # Save dataset
    filename = 'synthetic_test_data.csv'
    synthetic_df.to_csv(filename, index=False)
    print(f"\n💾 Saved to: {filename}")
    
    # Test with proper separation
    model, scaler = test_model_no_leakage(synthetic_df)
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE - NO DATA LEAKAGE")
    print("=" * 70)
    print("\n📝 Key Protections Applied:")
    print("   ✓ Split data BEFORE feature engineering")
    print("   ✓ Test set uses ONLY training statistics")
    print("   ✓ Separate feature engineers for train/test")
    print("   ✓ Scaler fitted on training data only")
    print("   ✓ SMOTE applied to training data only")
    
    return synthetic_df, model

if __name__ == "__main__":
    main()
