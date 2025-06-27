# main_app.py
# -----------------------------------------------------------------------------
# This script creates a Flask web server with two API endpoints to detect
# potential fraud in UPI (Unified Payments Interface) transactions.
#
# 1. /predict_text: Analyzes a string of text using a pre-trained NLP model
#    (DistilBERT) to assess the risk based on sentiment and tone.
# 2. /predict_transaction: Analyzes a structured set of transaction data
#    using a pre-trained scikit-learn model to predict if it is fraudulent.
#
# --- SETUP INSTRUCTIONS ---
# 1. Install dependencies:
#    pip install pandas flask flask_cors transformers torch scikit-learn
#
# 2. Place your trained model file in the same directory as this script.
#    This script expects it to be named 'all_models.pkl'.
#
# 3. Run the server:
#    python main_app.py
#
# 4. The server will start at http://127.0.0.1:5000. Open the HTML file
#    in your browser to interact with the application.
# -----------------------------------------------------------------------------

import pickle
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# --- 1. APPLICATION SETUP ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow the frontend HTML
# file to make requests to this server.
CORS(app)

# Configure logging to display informative messages in the console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. LOAD MODELS ON STARTUP ---
nlp_pipeline = None
ml_model = None
feature_names = None
MODEL_PATH = 'all_models.pkl'
NLP_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'

def load_nlp_pipeline():
    """Loads the Hugging Face Transformer pipeline for text analysis."""
    global nlp_pipeline
    try:
        logger.info(f"Loading NLP pipeline with model: {NLP_MODEL_NAME}")
        # Use GPU if available for faster processing, otherwise default to CPU.
        device = 0 if torch.cuda.is_available() else -1
        if device == -1:
            logger.warning("No GPU detected. Using CPU for the NLP model, which may be slower.")

        tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_NAME)
        nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        logger.info("✅ NLP pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load NLP pipeline: {str(e)}")
        nlp_pipeline = None

def load_ml_model():
    """Loads the trained scikit-learn model from the pickle file."""
    global ml_model, feature_names
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"⚠️ Model file '{MODEL_PATH}' not found. Transaction prediction will not be available.")
        return

    try:
        logger.info(f"Loading ML model from '{MODEL_PATH}'...")
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        
        # The model is stored in a dictionary; retrieve the ensemble model and feature names.
        ml_model = data['models'].get('Voting Ensemble') or data['models'].get('Stacked Ensemble')
        feature_names = data['feature_names']
        
        if ml_model is None:
            raise ValueError("No valid ensemble model found in the pickle file.")
            
        logger.info("✅ ML model and features loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load ML model: {str(e)}")
        ml_model = None
        feature_names = None

# Load all models when the application starts.
load_nlp_pipeline()
load_ml_model()

# --- 3. API ENDPOINTS ---

@app.route('/')
def status():
    """
    Root endpoint to check if the server is running and what models are loaded.
    This helps in debugging the connection from the frontend.
    """
    return jsonify({
        "status": "Server is running",
        "nlp_model_loaded": nlp_pipeline is not None,
        "transaction_model_loaded": ml_model is not None
    })

@app.route('/predict_text', methods=['POST'])
def predict_text():
    """Analyzes text from the user to determine scam risk."""
    if not nlp_pipeline:
        logger.error("NLP pipeline not available for /predict_text request.")
        return jsonify({'error': 'NLP model is not available on the server.'}), 500

    json_data = request.get_json()
    if not json_data or 'message' not in json_data or not json_data['message'].strip():
        logger.warning("Invalid input for /predict_text: 'message' field is missing or empty.")
        return jsonify({'error': 'Invalid input. A non-empty "message" field is required.'}), 400
    
    user_message = json_data['message']
    
    try:
        logger.info(f"Analyzing text: '{user_message}'")
        result = nlp_pipeline(user_message)[0]
        label = result['label']
        score = result['score']

        # Define rules for determining risk level based on sentiment.
        if label == 'NEGATIVE' and score > 0.9:
            risk_level = 'DANGER'
            advice = "This message shows strong signs of urgency and negativity, a common tactic in scams. **HIGH RISK**. Do not proceed. Cease communication and block the sender."
        elif label == 'NEGATIVE':
            risk_level = 'WARNING'
            advice = "The tone of this message is negative, which could indicate pressure or manipulation. Proceed with extreme caution. Verify payment requests through a trusted channel."
        else:
            risk_level = 'SAFE'
            advice = "The tone appears neutral. No immediate red flags, but always double-check recipient details before sending money."

        logger.info(f"Text analysis result: {risk_level}, Score: {score:.2f}")
        return jsonify({
            'riskLevel': risk_level,
            'advice': advice,
            'detectedFlags': [f"Tone: {label}", f"Confidence: {score:.2f}"]
        })

    except Exception as e:
        logger.error(f"Error during NLP analysis: {str(e)}")
        return jsonify({'error': f'Failed to analyze message: {str(e)}'}), 500

@app.route('/predict_transaction', methods=['POST'])
def predict_transaction():
    """Predicts fraud for a structured transaction."""
    if not ml_model or not feature_names:
        logger.error("ML model not available for /predict_transaction request.")
        return jsonify({'error': 'The transaction prediction model is not available.'}), 500

    json_data = request.get_json()
    if not json_data:
        logger.warning("Invalid input for /predict_transaction: JSON data is required.")
        return jsonify({'error': 'Invalid input. JSON data is required.'}), 400

    try:
        # Create a DataFrame from the incoming JSON.
        input_df = pd.DataFrame([json_data])
        
        # --- Feature Engineering ---
        # Recreate the same features used during model training.
        input_df['hour'] = input_df['step'] % 24
        input_df['amount_ratio'] = input_df['amount'] / (input_df['oldbalanceOrg'] + 1e-6)
        input_df = input_df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore')
        
        # --- Pre-processing ---
        # One-hot encode the 'type' column.
        processed_df = pd.get_dummies(input_df, columns=['type'], drop_first=True, dtype=int)
        # Align columns with the model's training features to prevent errors.
        processed_df = processed_df.reindex(columns=feature_names, fill_value=0)

        # --- Prediction ---
        prediction = ml_model.predict(processed_df)
        probability = ml_model.predict_proba(processed_df)[:, 1]

        logger.info(f"Transaction prediction: isFraud={prediction[0]}, Probability={probability[0]:.4f}")
        return jsonify({
            'isFraudPrediction': int(prediction[0]),
            'fraudProbability': float(probability[0])
        })

    except Exception as e:
        logger.error(f"Error during transaction prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    # Run the app with debug=True for development. This provides detailed error
    # messages and automatically reloads the server when you change the code.
    app.run(host='0.0.0.0', port=5000, debug=True)
