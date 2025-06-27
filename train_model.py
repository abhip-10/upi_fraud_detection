import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pickle
from imblearn.over_sampling import SMOTE

def create_and_save_all_models():
    print("Starting model training")
    dataset_path = 'Dataset.csv'
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found: {dataset_path}")
        return
    print(f"Original dataset size: {len(df)} rows")
    df_sample = df.sample(frac=0.5, random_state=42, stratify=df['isFraud'])
    print(f"Using sample of {len(df_sample)} rows")
    df_filtered = df_sample[df_sample['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    df_filtered['hour'] = df_filtered['step'] % 24
    df_filtered['amount_ratio'] = df_filtered['amount'] / (df_filtered['oldbalanceOrg'] + 1e-6)
    df_filtered['transaction_count'] = df_filtered.groupby('nameOrig')['step'].transform('count')
    X = df_filtered.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    y = df_filtered['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)
    X_train_processed = pd.get_dummies(X_train, columns=['type'], drop_first=True, dtype=int)
    feature_names = X_train_processed.columns.tolist()
    X_test_processed = pd.get_dummies(X_test, columns=['type'], drop_first=True, dtype=int)
    X_test_processed = X_test_processed.reindex(columns=feature_names, fill_value=0)
    smote = SMOTE(random_state=42)
    X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    models_to_save = {}
    evaluation_results = {}
    def evaluate_model(model_name, model, X_test_data, y_test_data):
        print(f"Evaluating {model_name}")
        y_pred = model.predict(X_test_data)
        y_proba = model.predict_proba(X_test_data)[:, 1]
        evaluation_results[model_name] = {
            'Accuracy': accuracy_score(y_test_data, y_pred),
            'Precision': precision_score(y_test_data, y_pred),
            'Recall': recall_score(y_test_data, y_pred),
            'F1-Score': f1_score(y_test_data, y_pred),
            'AUC-ROC': roc_auc_score(y_test_data, y_proba),
            'Classification Report': classification_report(y_test_data, y_pred),
            'Confusion Matrix': confusion_matrix(y_test_data, y_pred)
        }
    print("Training Logistic Regression")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    models_to_save['Logistic Regression'] = lr_model
    evaluate_model('Logistic Regression', lr_model, X_test_scaled, y_test)
    print("Training Random Forest")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    rf_model = GridSearchCV(RandomForestClassifier(class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1),
                            param_grid, cv=3, scoring='f1')
    rf_model.fit(X_train_processed, y_train)
    models_to_save['Random Forest'] = rf_model.best_estimator_
    evaluate_model('Random Forest', rf_model.best_estimator_, X_test_processed, y_test)
    print("Training XGBoost")
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1, scale_pos_weight=10)
    xgb_model.fit(X_train_processed, y_train)
    models_to_save['XGBoost'] = xgb_model
    evaluate_model('XGBoost', xgb_model, X_test_processed, y_test)
    print("Training LightGBM")
    lgb_model = lgb.LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_jobs=-1, class_weight={0: 1, 1: 10})
    lgb_model.fit(X_train_processed, y_train)
    models_to_save['LightGBM'] = lgb_model
    evaluate_model('LightGBM', lgb_model, X_test_processed, y_test)
    print("Training Stacked Ensemble")
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1)),
        ('lgbm', lgb.LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_jobs=-1))
    ]
    meta_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
    stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3, n_jobs=-1)
    stacked_model.fit(X_train_processed, y_train)
    models_to_save['Stacked Ensemble'] = stacked_model
    evaluate_model('Stacked Ensemble', stacked_model, X_test_processed, y_test)
    print("MODEL EVALUATION SUMMARY")
    for model_name, metrics in evaluation_results.items():
        print(f"Results for {model_name}")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
        print("Confusion Matrix:")
        print(metrics['Confusion Matrix'])
        print("Classification Report:")
        print(metrics['Classification Report'])
    data_to_save = {
        'models': models_to_save,
        'feature_names': feature_names,
        'scaler': scaler,
        'evaluation_results': evaluation_results,
        'metadata': {'training_date': '2025-06-27', 'sample_size': len(df_sample)}
    }
    with open('all_models.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    X_test.to_csv('test_data.csv', index=False)
    y_test.to_csv('test_labels.csv', index=False)
    print("All models trained and saved")

if __name__ == '__main__':
    create_and_save_all_models()
