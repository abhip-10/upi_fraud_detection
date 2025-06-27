import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
import pickle
import numpy as np

def create_and_save_all_models():
    """
    Loads a SAMPLE of data, preprocesses it, trains all specified models,
    evaluates them, and saves them to a single file for the app to use.
    """
    print("--- Starting Full Model Training, Evaluation & Preprocessing ---")

    # 1. Load and process the raw data
    dataset_path = 'Dataset.csv'
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset not found. Please place '{dataset_path}' in the project directory.")
        return

    # --- SPEED OPTIMIZATION: Train on a sample of the data ---
    print(f"Original dataset size: {len(df)} rows.")
    # Using a smaller sample for quick demonstration. Increase frac for better performance.
    df_sample = df.sample(frac=0.1, random_state=42)
    print(f"Using a sample of {len(df_sample)} rows for faster local training.")
    # ---------------------------------------------------------

    # 2. Feature Engineering and Filtering
    df_filtered = df_sample[df_sample['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    df_filtered['hour'] = df_filtered['step'] % 24
    df_filtered['amount_ratio'] = df_filtered['amount'] / (df_filtered['oldbalanceOrg'] + 1e-6)

    X = df_filtered.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    y = df_filtered['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y
    )

    # 3. Preprocess training data
    # One-hot encode categorical features for tree-based models
    X_train_processed = pd.get_dummies(X_train, columns=['type'], drop_first=True, dtype=int)
    feature_names = X_train_processed.columns.tolist()

    # Preprocess test data similarly
    X_test_processed = pd.get_dummies(X_test, columns=['type'], drop_first=True, dtype=int)
    # Ensure test columns match train columns
    X_test_processed = X_test_processed.reindex(columns=feature_names, fill_value=0)


    # Scale data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed) # Use the same scaler on test data

    # --- Train All Models & Evaluate ---
    models_to_save = {}
    evaluation_results = {}

    def evaluate_model(model_name, model, X_test_data, y_test_data):
        """Helper function to evaluate a model and store its metrics."""
        print(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test_data)
        y_proba = model.predict_proba(X_test_data)[:, 1]

        accuracy = accuracy_score(y_test_data, y_pred)
        precision = precision_score(y_test_data, y_pred)
        recall = recall_score(y_test_data, y_pred)
        f1 = f1_score(y_test_data, y_pred)
        auc = roc_auc_score(y_test_data, y_proba)
        report = classification_report(y_test_data, y_pred)
        cm = confusion_matrix(y_test_data, y_pred)

        evaluation_results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc,
            'Classification Report': report,
            'Confusion Matrix': cm
        }

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    models_to_save['Logistic Regression'] = (lr_model, scaler)
    evaluate_model('Logistic Regression', lr_model, X_test_scaled, y_test)

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_processed, y_train)
    models_to_save['Random Forest'] = rf_model
    evaluate_model('Random Forest', rf_model, X_test_processed, y_test)

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_processed, y_train)
    models_to_save['XGBoost'] = xgb_model
    evaluate_model('XGBoost', xgb_model, X_test_processed, y_test)

    # --- LightGBM ---
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_jobs=-1)
    lgb_model.fit(X_train_processed, y_train)
    models_to_save['LightGBM'] = lgb_model
    evaluate_model('LightGBM', lgb_model, X_test_processed, y_test)

    # --- Stacked Ensemble ---
    print("\nTraining Stacked Ensemble...")
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1)),
        ('lgbm', lgb.LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_jobs=-1))
    ]
    meta_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
    stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3, n_jobs=-1)
    stacked_model.fit(X_train_processed, y_train)
    models_to_save['Stacked Ensemble'] = stacked_model
    evaluate_model('Stacked Ensemble', stacked_model, X_test_processed, y_test)

    # 4. Display all evaluation results
    print("\n\n--- MODEL EVALUATION SUMMARY ---")
    for model_name, metrics in evaluation_results.items():
        print(f"\n---------------------------------")
        print(f"    Results for {model_name}")
        print(f"---------------------------------")
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"AUC-ROC:   {metrics['AUC-ROC']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['Confusion Matrix'])
        print("\nClassification Report:")
        print(metrics['Classification Report'])
    print("---------------------------------\n")

    # 5. Save all trained models and necessary objects to a single file
    data_to_save = {
        'models': models_to_save,
        'feature_names': feature_names,
        'evaluation_results': evaluation_results
    }

    with open('all_models.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    # 6. Save the test data for the app to use
    X_test.to_csv('test_data.csv', index=False)
    y_test.to_csv('test_labels.csv', index=False)

    print("\n--- All models trained, evaluated, and saved successfully! ---")
    print("Saved 'all_models.pkl', 'test_data.csv', and 'test_labels.csv' to the project folder.")

# --- Main execution block ---
if __name__ == '__main__':
    create_and_save_all_models()