import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re

app = Flask(__name__)
CORS(app)

sentiment_pipeline = None
tabular_model = None
scaler = None
feature_names = None

def load_models():
    global sentiment_pipeline, tabular_model, scaler, feature_names
    
    try:
        print("Loading Hugging Face model: ProsusAI/finbert...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert"
        )
        print("--- FinBERT model loaded successfully! ---")
    except Exception as e:
        print(f"--- FATAL: Could not load FinBERT model. Error: {e} ---")

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

def interpret_sentiment(sentiment_result, text):
    sentiment = sentiment_result[0]['label']
    score = sentiment_result[0]['score']
    text_lower = text.lower()
    
    positive_scam_keywords = [
        'won', 'lottery', 'claim', 'prize', 'cashback', 'winner', 'free giveaway',
        'congratulations you have won', 'exclusive reward', 'guaranteed profit',
        'investment opportunity', 'risk-free', 'refund approved', 'inheritance',
        'grant', 'stimulus', 'you are selected', 'special offer for you',
        'subsidy approved', 'job offer', 'easy loan', 'double your money',
        'government yojana', 'pm yojana benefit', 'gst refund', 'income tax refund',
        'kaun banega crorepati', 'kbc lottery', 'free recharge'
    ]
    
    neutral_scam_keywords = [
        'security deposit', 'verify your details', 'customs fee', 'pending bill',
        'update kyc', 'confirm your details', 'account verification', 'login attempt',
        'mandate', 'shipping fee', 'unlock', 'pending invoice', 'compliance check',
        'document verification', 'confirm identity', 'transaction pending',
        'background check fee', 'processing fee', 'registration fee', 'update pan card',
        'link aadhaar', 'e-kyc verification', 'upi pin', 'scan qr for payment',
        'block your sim', 'electricity bill will be disconnected', 'demat account',
        'your parcel is on hold', 'customs clearance', 'work from home'
    ]
    
    negative_scam_keywords = [
        'suspended', 'unusual activity', 'account blocked', 'action required',
        'security breach', 'account frozen', 'suspicious login', 'unauthorized transaction',
        'compromised', 'legal action', 'warrant', 'account will be terminated',
        'deactivation alert', 'immediate action required', 'fraud alert',
        'your account is blocked by rbi', 'police case filed', 'cbi notice',
        'illegal activity detected', 'your sim will be blocked', 
        'electricity connection will be disconnected'
    ]
    
    legitimate_negative_keywords = [
        'overdue', 'bill is due', 'late fee', 'payment reminder', 'low balance',
        'minimum balance', 'payment due', 'service charge', 'annual fee',
        'transaction declined', 'insufficient funds', 'credit limit reached',
        'emi due', 'automatic debit', 'upcoming payment', 'lic premium due',
        'fastag low balance', 'gas cylinder booking', 'electricity bill generated',
        'credit card statement', 'debit alert', 'fastag recharge'
    ]
    
    if sentiment == 'negative':
        if any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in legitimate_negative_keywords):
            return False, score
        is_fraud = True
        confidence = score + (1 - score) * 0.5
        return is_fraud, confidence

    if sentiment == 'positive':
        if any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in positive_scam_keywords):
            return True, 0.95
        return False, score

    if sentiment == 'neutral':
        if any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in neutral_scam_keywords):
            return True, 0.88
        return False, score
        
    return False, 0.0

@app.route('/predict', methods=['POST'])
def predict():
    if not sentiment_pipeline and not tabular_model:
        return jsonify({'error': 'No models are available. Check server logs.'}), 500

    try:
        data = request.get_json()
        message = data.get('textMessage')
        amount = float(data.get('amount') or 0)

        nlp_is_fraud, nlp_confidence = False, 0.0
        tabular_is_fraud, tabular_confidence = False, 0.0
        model_used = []

        if message and sentiment_pipeline:
            print(f"Analyzing text with FinBERT: '{message}'")
            model_used.append('FinBERT')
            sentiment_results = sentiment_pipeline(message)
            nlp_is_fraud, nlp_confidence = interpret_sentiment(sentiment_results, message)

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

        if 'FinBERT' in model_used and 'Tabular Model' in model_used:
            if nlp_is_fraud and nlp_confidence > 0.8:
                final_confidence = nlp_confidence
            else:
                final_confidence = (0.6 * nlp_confidence if nlp_is_fraud else (1-nlp_confidence)*-0.6) + \
                                     (0.4 * tabular_confidence)
            final_model_name = 'Hybrid'
        elif 'FinBERT' in model_used:
            final_confidence = nlp_confidence if nlp_is_fraud else (1-nlp_confidence)
            final_model_name = 'FinBERT'
        else:
            final_confidence = tabular_confidence
            final_model_name = 'Scikit-learn Tabular'
            
        is_fraud = final_confidence > 0.5

        response = {
            'isFraud': bool(is_fraud),
            'confidence': round(float(final_confidence if is_fraud else (1-final_confidence)) * 100, 2),
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
