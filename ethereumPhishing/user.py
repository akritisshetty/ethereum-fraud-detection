"""
user.py - Interactive User Interface for Ethereum Fraud Detection
Allows users to input transaction details and get real-time fraud predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from collections import defaultdict

# Import from fraud_detection.py
try:
    from fraud_detection import (
        EthereumFeatureEngineering,
        DataPreprocessor,
        FraudDetectionModel
    )
except ImportError:
    print("❌ ERROR: Cannot import from fraud_detection.py")
    print("   Make sure fraud_detection.py is in the same directory!")
    sys.exit(1)

print("\n" + "=" * 70)
print("  🔐 ETHEREUM FRAUD DETECTION - INTERACTIVE MODE")
print("=" * 70)

# ============================================================================
# MODEL LOADING & INITIALIZATION
# ============================================================================

class FraudDetectionSystem:
    """Complete fraud detection system with user interface"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.is_trained = False
        
    def train_or_load_model(self, csv_file):
        """Train a new model or load existing one"""
        
        print("\n📚 Initializing Fraud Detection System...")
        
        # Check if pre-trained model exists
        model_file = 'fraud_model.pkl'
        
        if os.path.exists(model_file):
            print(f"\n   Found existing model: {model_file}")
            choice = input("   Load existing model? (y/n): ").strip().lower()
            
            if choice == 'y':
                try:
                    with open(model_file, 'rb') as f:
                        saved_data = pickle.load(f)
                        self.model = saved_data['model']
                        self.preprocessor = saved_data['preprocessor']
                        self.feature_engineer = saved_data['feature_engineer']
                    print("   ✅ Model loaded successfully!")
                    self.is_trained = True
                    return
                except Exception as e:
                    print(f"   ⚠️  Error loading model: {e}")
                    print("   Will train new model instead...")
        
        # Train new model
        print("\n🤖 Training new model...")
        print(f"   Dataset: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"\n❌ ERROR: Dataset not found: {csv_file}")
            print("   Please ensure your CSV file is in the same directory.")
            sys.exit(1)
        
        try:
            # Initialize components
            self.feature_engineer = EthereumFeatureEngineering()
            self.preprocessor = DataPreprocessor()
            fraud_model = FraudDetectionModel()
            
            # Load data
            df = self.feature_engineer.load_and_analyze_data(csv_file)
            
            # Split BEFORE feature engineering
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df['Class']
            )
            
            print("   • Engineering features...")
            
            # Engineer features separately
            train_engineered = self.feature_engineer.engineer_features(
                train_df.reset_index(drop=True), is_training=True
            )
            
            # Prepare and scale
            X_train, y_train = self.preprocessor.prepare_features(train_engineered)
            
            from sklearn.preprocessing import StandardScaler
            self.preprocessor.scaler = StandardScaler()
            X_train_scaled = self.preprocessor.scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.preprocessor.feature_columns)
            
            # Balance with SMOTE
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            print("   • Training XGBoost model...")
            self.model = fraud_model.train_model(X_train_balanced, y_train_balanced)
            
            print("   ✅ Model trained successfully!")
            
            # Save model
            print("\n💾 Saving model...")
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'feature_engineer': self.feature_engineer
                }, f)
            print(f"   ✅ Model saved to: {model_file}")
            
            self.is_trained = True
            
        except Exception as e:
            print(f"\n❌ ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def predict_transaction(self, tx_data):
        """Predict fraud probability for a transaction"""
        
        if not self.is_trained:
            print("❌ Model not trained! Please train the model first.")
            return None, None
        
        try:
            # Convert to DataFrame
            tx_df = pd.DataFrame([tx_data])
            
            # Engineer basic features
            tx_df['TimeStamp'] = pd.to_datetime(tx_df['TimeStamp'], unit='s')
            tx_df['is_dust'] = (tx_df['Value'] < 0.001).astype(int)
            tx_df['value_log'] = np.log1p(tx_df['Value'])
            tx_df['is_zero_value'] = (tx_df['Value'] == 0).astype(int)
            
            # Time features
            tx_df['hour_of_day'] = tx_df['TimeStamp'].dt.hour
            tx_df['day_of_week'] = tx_df['TimeStamp'].dt.dayofweek
            tx_df['is_weekend'] = (tx_df['day_of_week'] >= 5).astype(int)
            
            # Apply stored frequency encodings
            tx_df['from_address_freq'] = tx_df['From'].map(
                self.feature_engineer.address_freq_from).fillna(1)
            tx_df['to_address_freq'] = tx_df['To'].map(
                self.feature_engineer.address_freq_to).fillna(1)
            
            # Apply fraud rates
            tx_df['from_fraud_rate'] = tx_df['From'].map(
                self.feature_engineer.from_fraud_rates).fillna(0.5)
            tx_df['to_fraud_rate'] = tx_df['To'].map(
                self.feature_engineer.to_fraud_rates).fillna(0.5)
            
            # Contract features
            tx_df['has_contract_address'] = (tx_df.get('ContractAddress', pd.Series([None])).notna()).astype(int)
            tx_df['has_input_data'] = (tx_df.get('Input', pd.Series(['0x']))[0] != '0x')
            tx_df['input_length'] = len(tx_df.get('Input', pd.Series(['0x']))[0])
            
            # Extract features
            features = []
            for col in self.preprocessor.feature_columns:
                if col in tx_df.columns:
                    features.append(tx_df[col].values[0])
                else:
                    # Default values for missing features
                    default_values = {
                        'time_delta_seconds': 0,
                        'is_new_recipient': 1,
                        'tx_sequence': 1,
                        'tx_count_1h': 1,
                        'from_avg_value': tx_data.get('Value', 0),
                        'value_deviation': 0
                    }
                    features.append(default_values.get(col, 0))
            
            # Scale features
            features_scaled = self.preprocessor.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0, 1]
            
            return prediction, probability
            
        except Exception as e:
            print(f"\n❌ ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None

# ============================================================================
# USER INPUT INTERFACE
# ============================================================================

def validate_eth_address(address):
    """Validate Ethereum address format"""
    if not address.startswith('0x'):
        return False
    if len(address) != 42:
        return False
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False

def validate_tx_hash(tx_hash):
    """Validate transaction hash format"""
    if not tx_hash.startswith('0x'):
        return False
    if len(tx_hash) != 66:
        return False
    try:
        int(tx_hash[2:], 16)
        return True
    except ValueError:
        return False

def get_user_transaction():
    """Get transaction details from user input"""
    
    print("\n" + "=" * 70)
    print("  📝 ENTER TRANSACTION DETAILS")
    print("=" * 70)
    
    transaction = {}
    
    # Transaction Hash
    while True:
        tx_hash = input("\n1️⃣  Transaction Hash (0x...): ").strip()
        if not tx_hash:
            tx_hash = '0x' + 'a' * 64  # Default
            print(f"   Using default: {tx_hash[:20]}...")
            break
        if validate_tx_hash(tx_hash):
            break
        print("   ⚠️  Invalid format! Must be 0x followed by 64 hex characters")
    transaction['TxHash'] = tx_hash
    
    # Block Height
    while True:
        block = input("\n2️⃣  Block Height (e.g., 18000000): ").strip()
        if not block:
            block = '18000000'
            print(f"   Using default: {block}")
            break
        try:
            block = int(block)
            if block > 0:
                break
            print("   ⚠️  Must be a positive number!")
        except ValueError:
            print("   ⚠️  Invalid number!")
    transaction['BlockHeight'] = int(block)
    
    # Timestamp
    while True:
        print("\n3️⃣  Timestamp:")
        print("   a) Use current time")
        print("   b) Enter Unix timestamp")
        print("   c) Enter date (YYYY-MM-DD)")
        choice = input("   Choice (a/b/c): ").strip().lower()
        
        if choice == 'a' or not choice:
            timestamp = int(datetime.now().timestamp())
            print(f"   Using current time: {datetime.fromtimestamp(timestamp)}")
            break
        elif choice == 'b':
            ts = input("   Unix timestamp: ").strip()
            try:
                timestamp = int(ts)
                print(f"   Date: {datetime.fromtimestamp(timestamp)}")
                break
            except ValueError:
                print("   ⚠️  Invalid timestamp!")
        elif choice == 'c':
            date_str = input("   Date (YYYY-MM-DD): ").strip()
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                timestamp = int(dt.timestamp())
                print(f"   Using: {dt}")
                break
            except ValueError:
                print("   ⚠️  Invalid date format!")
    transaction['TimeStamp'] = timestamp
    
    # From Address
    while True:
        from_addr = input("\n4️⃣  From Address (0x...): ").strip()
        if not from_addr:
            from_addr = '0x' + 'b' * 40
            print(f"   Using default: {from_addr[:20]}...")
            break
        if validate_eth_address(from_addr):
            break
        print("   ⚠️  Invalid format! Must be 0x followed by 40 hex characters")
    transaction['From'] = from_addr
    
    # To Address
    while True:
        to_addr = input("\n5️⃣  To Address (0x...): ").strip()
        if not to_addr:
            to_addr = '0x' + 'c' * 40
            print(f"   Using default: {to_addr[:20]}...")
            break
        if validate_eth_address(to_addr):
            break
        print("   ⚠️  Invalid format! Must be 0x followed by 40 hex characters")
    transaction['To'] = to_addr
    
    # Value
    while True:
        value = input("\n6️⃣  Value in ETH (e.g., 0.5 or 0.0001): ").strip()
        if not value:
            value = '1.0'
            print(f"   Using default: {value} ETH")
            break
        try:
            value = float(value)
            if value >= 0:
                break
            print("   ⚠️  Must be non-negative!")
        except ValueError:
            print("   ⚠️  Invalid number!")
    transaction['Value'] = float(value)
    
    # Contract Address (optional)
    contract = input("\n7️⃣  Contract Address (press Enter to skip): ").strip()
    if contract and validate_eth_address(contract):
        transaction['ContractAddress'] = contract
    else:
        transaction['ContractAddress'] = None
    
    # Input Data
    input_data = input("\n8️⃣  Input Data (press Enter for '0x'): ").strip()
    transaction['Input'] = input_data if input_data else '0x'
    
    return transaction

def display_prediction_result(tx_data, prediction, probability):
    """Display fraud prediction results"""
    
    print("\n" + "=" * 70)
    print("  🎯 FRAUD DETECTION RESULT")
    print("=" * 70)
    
    # Risk assessment
    if probability >= 0.9:
        risk_level = "🔴 CRITICAL"
        risk_color = "\033[91m"  # Red
    elif probability >= 0.7:
        risk_level = "🟠 HIGH"
        risk_color = "\033[93m"  # Yellow
    elif probability >= 0.5:
        risk_level = "🟡 MODERATE"
        risk_color = "\033[93m"  # Yellow
    elif probability >= 0.3:
        risk_level = "🟢 LOW"
        risk_color = "\033[92m"  # Green
    else:
        risk_level = "✅ SAFE"
        risk_color = "\033[92m"  # Green
    reset_color = "\033[0m"
    
    print(f"\n  {risk_color}Risk Level: {risk_level}{reset_color}")
    print(f"  Fraud Probability: {probability:.2%}")
    print(f"  Prediction: {'🚨 FRAUD' if prediction == 1 else '✅ LEGITIMATE'}")
    
    print("\n  📋 Transaction Summary:")
    print(f"     Hash:  {tx_data['TxHash'][:20]}...")
    print(f"     From:  {tx_data['From'][:20]}...")
    print(f"     To:    {tx_data['To'][:20]}...")
    print(f"     Value: {tx_data['Value']} ETH")
    print(f"     Time:  {datetime.fromtimestamp(tx_data['TimeStamp'])}")
    
    # Warning indicators
    if tx_data['Value'] < 0.001:
        print("\n  ⚠️  WARNING INDICATORS:")
        print(f"     🔸 DUST TRANSACTION: Value is extremely low ({tx_data['Value']} ETH)")
        print(f"        → Common in address poisoning attacks")
    
    if probability >= 0.7:
        print("\n  🛡️  SECURITY RECOMMENDATIONS:")
        print("     1. DO NOT proceed with this transaction")
        print("     2. Verify the recipient address carefully")
        print("     3. Check if address matches your intended recipient")
        print("     4. Be aware of address poisoning scams")
        print("     5. Use your wallet's address book feature")
    elif probability >= 0.5:
        print("\n  ⚠️  CAUTION ADVISED:")
        print("     1. Double-check recipient address")
        print("     2. Verify transaction details")
        print("     3. Proceed with caution")
    else:
        print("\n  ✅ Transaction appears legitimate")
    
    print("=" * 70)

def interactive_mode(system):
    """Run interactive prediction mode"""
    
    print("\n" + "=" * 70)
    print("  🔄 INTERACTIVE FRAUD DETECTION MODE")
    print("=" * 70)
    print("\n  You can now check multiple transactions for fraud.")
    print("  Enter transaction details to get real-time predictions.")
    
    transaction_count = 0
    
    while True:
        transaction_count += 1
        print(f"\n\n{'='*70}")
        print(f"  TRANSACTION #{transaction_count}")
        
        # Get transaction from user
        tx_data = get_user_transaction()
        
        print("\n⏳ Analyzing transaction...")
        
        # Predict
        prediction, probability = system.predict_transaction(tx_data)
        
        if prediction is not None:
            # Display results
            display_prediction_result(tx_data, prediction, probability)
        else:
            print("\n❌ Failed to analyze transaction.")
        
        # Continue?
        print("\n" + "=" * 70)
        another = input("\n  Check another transaction? (y/n): ").strip().lower()
        if another != 'y':
            break
    
    print("\n" + "=" * 70)
    print(f"  ✅ Analyzed {transaction_count} transaction(s)")
    print("  Thank you for using Ethereum Fraud Detection System!")
    print("=" * 70)

# ============================================================================
# BATCH MODE - Process CSV File
# ============================================================================

def batch_mode(system):
    """Process multiple transactions from CSV file"""
    
    print("\n" + "=" * 70)
    print("  📄 BATCH PROCESSING MODE")
    print("=" * 70)
    
    csv_file = input("\n  Enter CSV file path: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"  ❌ File not found: {csv_file}")
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"\n  ✅ Loaded {len(df)} transactions")
        
        results = []
        
        print("\n  ⏳ Analyzing transactions...")
        for idx, row in df.iterrows():
            tx_data = row.to_dict()
            prediction, probability = system.predict_transaction(tx_data)
            
            results.append({
                'TxHash': tx_data.get('TxHash', 'N/A'),
                'Prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                'Probability': probability,
                'Risk': 'HIGH' if probability >= 0.7 else 'MEDIUM' if probability >= 0.5 else 'LOW'
            })
            
            if (idx + 1) % 10 == 0:
                print(f"     Processed {idx + 1}/{len(df)} transactions...")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = 'fraud_detection_results.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"\n  ✅ Analysis complete!")
        print(f"  💾 Results saved to: {output_file}")
        
        # Summary
        fraud_count = (results_df['Prediction'] == 'FRAUD').sum()
        high_risk = (results_df['Risk'] == 'HIGH').sum()
        
        print(f"\n  📊 Summary:")
        print(f"     Total transactions: {len(results_df)}")
        print(f"     Flagged as fraud: {fraud_count} ({fraud_count/len(results_df)*100:.1f}%)")
        print(f"     High risk: {high_risk}")
        
    except Exception as e:
        print(f"\n  ❌ Error processing file: {e}")

# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize system
    system = FraudDetectionSystem()
    
    # Train or load model
    csv_file = '1st dataset - balanced.csv'
    system.train_or_load_model(csv_file)
    
    # Main menu
    while True:
        print("\n" + "=" * 70)
        print("  MAIN MENU")
        print("=" * 70)
        print("\n  1. Interactive Mode - Check individual transactions")
        print("  2. Batch Mode - Process CSV file")
        print("  3. Exit")
        
        choice = input("\n  Select option (1/2/3): ").strip()
        
        if choice == '1':
            interactive_mode(system)
        elif choice == '2':
            batch_mode(system)
        elif choice == '3':
            print("\n  👋 Goodbye!")
            break
        else:
            print("\n  ⚠️  Invalid option!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user")
        print("  👋 Goodbye!")
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
