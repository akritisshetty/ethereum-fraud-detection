"""
fraud_detection.py - Ethereum Address Poisoning Detection (NO DATA LEAKAGE)
Clean version with minimal output
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, auc, f1_score,
                             precision_score, recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# FEATURE ENGINEERING - FIXED FOR NO LEAKAGE
# ============================================================================

class EthereumFeatureEngineering:
    """Feature engineering with proper train/test separation"""
    
    def __init__(self):
        self.to_address_seen = defaultdict(set)
        self.address_freq_from = {}
        self.address_freq_to = {}
        self.from_fraud_rates = {}
        self.to_fraud_rates = {}
        
    def load_and_analyze_data(self, filepath):
        """Load CSV and perform initial analysis"""
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        print(f"\nDataset loaded: {df.shape[0]:,} transactions")
        print(f"Normal: {(df['Class']==0).sum():,} | Phishing: {(df['Class']==1).sum():,}")
        
        return df
    
    def fit_target_encoding(self, df):
        """Fit target encoding on training data only"""
        self.from_fraud_rates = df.groupby('From')['Class'].mean().to_dict()
        self.to_fraud_rates = df.groupby('To')['Class'].mean().to_dict()
        return self
    
    def engineer_features(self, df, is_training=True):
        """Engineer features WITHOUT data leakage"""
        df = df.copy()
        
        # Basic cleaning
        df['To'] = df['To'].fillna('CONTRACT_CREATION')
        df['Input'] = df['Input'].fillna('0x')
        df['Class'] = df['Class'].fillna(0)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], unit='s')
        df = df.sort_values('TimeStamp').reset_index(drop=True)
        
        # Dust detection
        df['is_dust'] = (df['Value'] < 0.001).astype(int)
        df['value_log'] = np.log1p(df['Value'])
        df['is_zero_value'] = (df['Value'] == 0).astype(int)
        
        # Address frequency
        if is_training:
            from_counts = df['From'].value_counts()
            to_counts = df['To'].value_counts()
            self.address_freq_from = from_counts.to_dict()
            self.address_freq_to = to_counts.to_dict()
        
        df['from_address_freq'] = df['From'].map(self.address_freq_from).fillna(1)
        df['to_address_freq'] = df['To'].map(self.address_freq_to).fillna(1)
        
        # Time features
        df['time_delta_seconds'] = df.groupby('From')['TimeStamp'].diff().dt.total_seconds()
        df['time_delta_seconds'] = df['time_delta_seconds'].fillna(0)
        df['hour_of_day'] = df['TimeStamp'].dt.hour
        df['day_of_week'] = df['TimeStamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # New recipient
        if is_training:
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
        
        # Contract interaction
        df['has_contract_address'] = (df['ContractAddress'].notna()).astype(int)
        df['has_input_data'] = (df['Input'] != '0x').astype(int)
        df['input_length'] = df['Input'].str.len()
        
        # Target encoding
        if is_training:
            self.fit_target_encoding(df)
        
        df['from_fraud_rate'] = df['From'].map(self.from_fraud_rates).fillna(0.5)
        df['to_fraud_rate'] = df['To'].map(self.to_fraud_rates).fillna(0.5)
        
        # Statistical features
        df['from_avg_value'] = df.groupby('From')['Value'].transform('mean')
        df['value_deviation'] = abs(df['Value'] - df['from_avg_value'])
        
        return df

# ============================================================================
# PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Preprocessing with proper train/test separation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Select and prepare features"""
        exclude_cols = ['TxHash', 'BlockHeight', 'TimeStamp', 'From', 'To', 
                       'ContractAddress', 'Input', 'Class']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].fillna(0)
        y = df['Class']
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """Scale features - fit on train, transform both"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to training data only"""
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        return X_train_balanced, y_train_balanced

# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

class FraudDetectionModel:
    """XGBoost fraud detection model"""
    
    def __init__(self):
        self.model = None
        
    def train_model(self, X_train, y_train):
        """Train XGBoost classifier"""
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            tree_method='hist'
        )
        
        self.model.fit(X_train, y_train, verbose=False)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'cm': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_feature_importance(self, feature_names, top_n=15):
        """Visualize feature importance"""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top Feature Importance for Fraud Detection', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        
        return feature_importance_df

# ============================================================================
# REAL-TIME DETECTION
# ============================================================================

class SocialEngineeringDetector:
    """Real-time fraud detection"""
    
    def __init__(self, model, preprocessor, feature_engineer):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.alert_threshold = 0.7
        
    def predict_transaction(self, transaction_data):
        """Predict if transaction is fraudulent"""
        tx_df = pd.DataFrame([transaction_data])
        
        tx_df['TimeStamp'] = pd.to_datetime(tx_df['TimeStamp'], unit='s')
        tx_df['is_dust'] = (tx_df['Value'] < 0.001).astype(int)
        tx_df['value_log'] = np.log1p(tx_df['Value'])
        tx_df['is_zero_value'] = (tx_df['Value'] == 0).astype(int)
        
        tx_df['from_address_freq'] = tx_df['From'].map(
            self.feature_engineer.address_freq_from).fillna(1)
        tx_df['to_address_freq'] = tx_df['To'].map(
            self.feature_engineer.address_freq_to).fillna(1)
        
        features = []
        for col in self.preprocessor.feature_columns:
            if col in tx_df.columns:
                features.append(tx_df[col].values[0])
            else:
                features.append(0)
        
        features_scaled = self.preprocessor.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        return prediction, probability
    
    def generate_alert(self, transaction_data, probability):
        """Generate security alert"""
        print("\n" + "=" * 70)
        print("SECURITY ALERT: POTENTIAL FRAUD DETECTED")
        print("=" * 70)
        print(f"Fraud Probability: {probability:.2%}")
        print(f"Hash:  {transaction_data.get('TxHash', 'N/A')[:20]}...")
        print(f"Value: {transaction_data.get('Value', 0)} ETH")
        
        if transaction_data.get('Value', 0) < 0.001:
            print("WARNING: Dust transaction detected (address poisoning risk)")
        print("=" * 70)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute fraud detection pipeline"""
    
    print("\n" + "=" * 70)
    print("ETHEREUM ADDRESS POISONING DETECTION SYSTEM")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(script_dir, '1st dataset - balanced.csv')
    
    try:
        # Load data
        feature_engineer = EthereumFeatureEngineering()
        df = feature_engineer.load_and_analyze_data(CSV_FILE)
        
        # Split BEFORE feature engineering
        print("\nSplitting data (80/20)...")
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['Class']
        )
        print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
        
        # Engineer features separately
        print("Engineering features...")
        train_engineered = feature_engineer.engineer_features(
            train_df.reset_index(drop=True), is_training=True
        )
        
        test_engineer = EthereumFeatureEngineering()
        test_engineer.address_freq_from = feature_engineer.address_freq_from
        test_engineer.address_freq_to = feature_engineer.address_freq_to
        test_engineer.from_fraud_rates = feature_engineer.from_fraud_rates
        test_engineer.to_fraud_rates = feature_engineer.to_fraud_rates
        
        test_engineered = test_engineer.engineer_features(
            test_df.reset_index(drop=True), is_training=False
        )
        
        # Prepare features
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.prepare_features(train_engineered)
        X_test, y_test = preprocessor.prepare_features(test_engineered)
        
        # Scale
        print("Scaling features...")
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Balance
        print("Balancing with SMOTE...")
        X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
            X_train_scaled, y_train
        )
        
        # Train
        print("Training XGBoost model...")
        fraud_model = FraudDetectionModel()
        model = fraud_model.train_model(X_train_balanced, y_train_balanced)
        print("Training complete")
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = fraud_model.evaluate_model(X_test_scaled, y_test)
        cm = metrics['cm']
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE")
        print("=" * 70)
        print(f"Precision-Recall AUC: {metrics['pr_auc']:.4f}")
        print(f"F1-Score:             {metrics['f1']:.4f}")
        print(f"Precision:            {metrics['precision']:.4f}")
        print(f"Recall:               {metrics['recall']:.4f}")
        print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        print("=" * 70)
        
        # Feature importance
        print("\nTop 5 Features:")
        feature_importance = fraud_model.plot_feature_importance(
            preprocessor.feature_columns
        )
        for i, row in feature_importance.head(5).iterrows():
            print(f"{i+1}. {row['Feature']:30s} {row['Importance']:.4f}")
        
        print("\nFeature importance saved: feature_importance.png")
        
        # Demo
        print("\n" + "=" * 70)
        print("REAL-TIME DETECTION DEMO")
        print("=" * 70)
        
        detector = SocialEngineeringDetector(model, preprocessor, feature_engineer)
        sample = test_engineered.sample(min(2, len(test_engineered)))
        
        for _, tx in sample.iterrows():
            tx_data = tx.to_dict()
            prediction, probability = detector.predict_transaction(tx_data)
            
            if probability >= detector.alert_threshold:
                detector.generate_alert(tx_data, probability)
            else:
                print(f"\nTransaction {tx_data.get('TxHash', 'N/A')[:16]}... is SAFE (Risk: {probability:.2%})")
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find '{CSV_FILE}'")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
