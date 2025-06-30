import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import zipfile
import shutil

app = Flask(__name__)
CORS(app)

sentiment_pipeline = None
tabular_model = None
scaler = None
feature_names = None

FINBERT_MODEL_ZIP = 'finbert_fraud_model.zip'
FINBERT_MODEL_EXTRACT_DIR = './finbert_fraud_model_extracted'

def load_models():
    global sentiment_pipeline, tabular_model, scaler, feature_names
    
    model_loaded_from_fine_tuned = False
    finetuned_tokenizer = None
    finetuned_model = None

    if os.path.exists(FINBERT_MODEL_ZIP):
        print(f"Found fine-tuned FinBERT model zip: {FINBERT_MODEL_ZIP}. Attempting to extract and load...")
        try:
            if os.path.exists(FINBERT_MODEL_EXTRACT_DIR):
                print(f"Removing existing extraction directory: {FINBERT_MODEL_EXTRACT_DIR}")
                shutil.rmtree(FINBERT_MODEL_EXTRACT_DIR)
            
            with zipfile.ZipFile(FINBERT_MODEL_ZIP, 'r') as zip_ref:
                zip_ref.extractall(FINBERT_MODEL_EXTRACT_DIR)
            print(f"Fine-tuned model extracted to {FINBERT_MODEL_EXTRACT_DIR}")

            has_model_config = os.path.exists(os.path.join(FINBERT_MODEL_EXTRACT_DIR, "config.json"))
            has_model_weights = os.path.exists(os.path.join(FINBERT_MODEL_EXTRACT_DIR, "pytorch_model.bin")) or \
                                os.path.exists(os.path.join(FINBERT_MODEL_EXTRACT_DIR, "model.safetensors"))
            
            if has_model_config and has_model_weights:
                try:
                    finetuned_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_EXTRACT_DIR)
                    print("Tokenizer loaded from fine-tuned model directory.")
                except Exception as tokenizer_e:
                    print(f"WARNING: Could not load tokenizer from fine-tuned directory ({FINBERT_MODEL_EXTRACT_DIR}). Error: {tokenizer_e}")
                    print("This likely means tokenizer files (e.g., tokenizer_config.json, vocab.txt) are missing in the zip.")
                    print("Falling back to loading tokenizer from base model 'ProsusAI/finbert'.")
                    finetuned_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                    
                finetuned_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_EXTRACT_DIR)
                model_loaded_from_fine_tuned = True
                
                sentiment_pipeline = pipeline(
                    "text-classification",
                    model=finetuned_model,
                    tokenizer=finetuned_tokenizer 
                )
                print("--- Fine-tuned FinBERT model loaded successfully (possibly with base tokenizer)! ---")
            else:
                print(f"WARNING: Missing config.json or model weights (pytorch_model.bin/model.safetensors) in '{FINBERT_MODEL_EXTRACT_DIR}'.")
                print("Falling back to generic FinBERT model.")
        except Exception as e:
            print(f"--- FATAL: Could not extract or load fine-tuned FinBERT model. Error: {e} ---")
            print("--- Falling back to generic FinBERT model. ---")

    if not model_loaded_from_fine_tuned:
        print("Loading generic FinBERT model as primary (fine-tuned model not available or failed to load).")
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
            print("--- Generic FinBERT model loaded successfully! ---")
        except Exception as e:
            print(f"--- FATAL: Could not load generic FinBERT model. Error: {e} ---")

    try:
        print("Loading scikit-learn tabular model from 'all_models.pkl'...")
        with open('all_models.pkl', 'rb') as f:
            data = pickle.load(f)
            tabular_model = data['models']['Stacked Ensemble']
            scaler = data['scaler']
            feature_names = data['feature_names']
        print("--- Scikit-learn tabular model loaded successfully! ---")
    except FileNotFoundError:
        print("--- WARNING: 'all_models.pkl' not found. Full transaction analysis will be disabled. ---")
    except Exception as e:
        print(f"--- FATAL: Could not load scikit-learn model. Error: {e} ---")

def interpret_sentiment(sentiment_results):
    result = sentiment_results[0]
    
    if result['label'] == 'LABEL_1':
        is_fraud_decision = True
        prob_fraud = result['score']
    elif result['label'] == 'LABEL_0':
        is_fraud_decision = False
        prob_fraud = 1 - result['score'] 
    else:
        print(f"Warning: Unexpected label '{result['label']}' from sentiment pipeline. Interpreting generic FinBERT sentiment.")
        try:
            text_message_from_request = request.get_json().get('textMessage', '').lower()
        except RuntimeError:
            text_message_from_request = ""

        positive_scam_keywords = ['won', 'lottery', 'claim', 'prize', 'cashback', 'winner', 'free giveaway']
        neutral_scam_keywords = ['security deposit', 'verify your details', 'update kyc', 'confirm your details']
        negative_scam_keywords = ['suspended', 'unusual activity', 'account blocked', 'action required']
        
        if result['label'] == 'negative':
            if any(re.search(r'\b' + keyword + r'\b', text_message_from_request) for keyword in negative_scam_keywords):
                is_fraud_decision = True
                prob_fraud = 0.9
            else:
                is_fraud_decision = False
                prob_fraud = 0.2
        elif result['label'] == 'positive':
            if any(re.search(r'\b' + keyword + r'\b', text_message_from_request) for keyword in positive_scam_keywords):
                is_fraud_decision = True
                prob_fraud = 0.95
            else:
                is_fraud_decision = False
                prob_fraud = 0.1
        elif result['label'] == 'neutral':
            if any(re.search(r'\b' + keyword + r'\b', text_message_from_request) for keyword in neutral_scam_keywords):
                is_fraud_decision = True
                prob_fraud = 0.88 
            else:
                is_fraud_decision = False
                prob_fraud = 0.3 
        else: 
            is_fraud_decision = False
            prob_fraud = 0.5
            
    return is_fraud_decision, prob_fraud

@app.route('/predict', methods=['POST'])
def predict():
    if not sentiment_pipeline and not tabular_model:
        return jsonify({'error': 'No models are available. Check server logs.'}), 500

    try:
        data = request.get_json()
        message = data.get('textMessage')
        amount = float(data.get('amount') or 0)

        nlp_is_fraud_flag, nlp_prob_fraud = False, 0.0 
        tabular_is_fraud, tabular_confidence = False, 0.0
        model_used = []

        if message and sentiment_pipeline:
            print(f"Analyzing text with FinBERT: '{message}'")
            model_used.append('FinBERT')
            sentiment_results = sentiment_pipeline(message)
            nlp_is_fraud_flag, nlp_prob_fraud = interpret_sentiment(sentiment_results)

        if amount > 0 and tabular_model:
            print(f"Analyzing transaction details. Amount: {amount}")
            model_used.append('Tabular Model')
            
           
            required_fields = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
            for field in required_fields:
                if field == 'type': data[field] = data.get(field, 'TRANSFER')
                else: data[field] = data.get(field) or 0
            
            df = pd.DataFrame([data])
            df['hour'] = pd.to_numeric(df['step']) % 24
            df['amount_ratio'] = pd.to_numeric(df['amount']) / (pd.to_numeric(df['oldbalanceOrg']) + 1e-6)
            df['transaction_count'] = 1 
            
            
            df_processed = pd.get_dummies(df, columns=['type'], drop_first=True, dtype=int)
            df_processed = df_processed.reindex(columns=feature_names, fill_value=0)
            df_processed = df_processed[feature_names] 
            
            X_scaled = scaler.transform(df_processed)
            tabular_confidence = tabular_model.predict_proba(X_scaled)[0][1] 
            tabular_is_fraud = tabular_confidence > 0.5
            
        if not model_used:
            return jsonify({'error': 'Input not suitable for any available model. Please provide a message or transaction details.'}), 400

        
        final_confidence = 0.0
        final_model_name = ""

        if 'FinBERT' in model_used and 'Tabular Model' in model_used:
        
            final_confidence = (0.6 * nlp_prob_fraud if nlp_is_fraud_flag else (1 - nlp_prob_fraud) * -0.6) + \
                               (0.4 * tabular_confidence)
            final_model_name = 'Hybrid'
        elif 'FinBERT' in model_used:
            final_confidence = nlp_prob_fraud
            final_model_name = 'FinBERT'
        else: 
            final_confidence = tabular_confidence
            final_model_name = 'Scikit-learn Tabular'
            
        is_fraud_final = final_confidence > 0.5 
        
    
        output_confidence_percentage = round(float(final_confidence if is_fraud_final else (1 - final_confidence)) * 100, 2)

        response = {
            'isFraud': bool(is_fraud_final),
            'confidence': output_confidence_percentage,
            'modelUsed': final_model_name
        }

        print(f"Prediction result: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred.'}), 400

if __name__ == '__main__':
    load_models() 
    if sentiment_pipeline or tabular_model:
        print("Starting Flask server...")
    
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Flask server not started due to model loading failures.")
