
---

# 💳 UPI Fraud Detection System

A hybrid system to detect fraudulent financial activity through both:

* **NLP-based SMS/message classification** using FinBERT (Transformer-based model fine-tuned on scam messages)
* **Tabular transaction classification** using optimized machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM, Stacked Ensemble)

## 🔧 Features

* ✅ Detect fraud from transaction messages using `FinBERT`
* ✅ Analyze transaction metadata (amount, balance, type) using trained ML models
* ✅ Hybrid inference logic combining both models when available
* ✅ REST API with Flask for integration with external systems
* ✅ Interactive console & Streamlit compatibility
* ✅ Automatically loads and extracts fine-tuned models if available

---

## 📁 Project Structure

```
.
├── app.py                        # Flask backend
├── all_models.pkl               # Serialized traditional ML models and metadata
├── finbert_fraud_model.zip      # Zipped fine-tuned FinBERT model
├── test_data.csv                # Test set features (for tabular models)
├── test_labels.csv              # Corresponding labels
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Models Used

### 1. **FinBERT (Text Classifier)**

* Fine-tuned on synthetic fraud and legit message templates
* Uses HuggingFace Transformers
* If fine-tuned model zip is missing, defaults to generic `ProsusAI/finbert`

### 2. **Tabular Model (Transaction Classifier)**

* Trained on transaction data (like `Dataset.csv`)
* Features include `amount`, `balance`, `type`, `hour`, etc.
* Ensemble of:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * LightGBM
  * Stacked Meta-Classifier (final output)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/upi-fraud-detection.git
cd upi-fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Sample `requirements.txt`:

```txt
flask
flask-cors
transformers
scikit-learn
pandas
numpy
xgboost
lightgbm
imbalanced-learn
```

### 3. Place Required Files

Ensure the following files are in the project root:

* `finbert_fraud_model.zip` (fine-tuned FinBERT model)
* `all_models.pkl` (trained tabular models)
* `test_data.csv` & `test_labels.csv` (for reference/testing)

### 4. Run the Flask Server

```bash
python app.py
```

Once the server starts, it will be accessible at:

```
http://localhost:5000
```

---

## 📡 API Usage

### **Endpoint**: `POST /predict`

### 🔻 Request Payload

You can send either just the message, just transaction data, or both.

```json
{
  "textMessage": "Your bank KYC has expired. Please update it using this link.",
  "amount": 10000,
  "step": 120,
  "oldbalanceOrg": 15000,
  "newbalanceOrig": 5000,
  "oldbalanceDest": 3000,
  "newbalanceDest": 13000,
  "type": "TRANSFER"
}
```

### 🔺 Response Example

```json
{
  "isFraud": true,
  "confidence": 92.53,
  "modelUsed": "Hybrid"
}
```

* `isFraud`: `true` if the message or transaction is predicted as fraudulent
* `confidence`: confidence in prediction (percentage)
* `modelUsed`: `"FinBERT"`, `"Scikit-learn Tabular"`, or `"Hybrid"`

---

## 📊 Training the Models

* Use your own UPI fraud dataset or synthetic generation as shown in the training script
* Save FinBERT model using HuggingFace `Trainer`
* Save tabular models using `pickle.dump()` as in `all_models.pkl`

---

## 🛡️ Model Fallback Logic

* If fine-tuned FinBERT fails to load, falls back to generic sentiment-based FinBERT
* If tabular model not found, only NLP is used
* Combines both for stronger inference when possible

---

## 🎯 Future Improvements

* 🔒 OTP/link validator using regex patterns
* 🗃️ Database integration for fraud case reporting
* 📲 WhatsApp or SMS API integration

---

## 📜 License

MIT License — Feel free to use, modify, and contribute!

---
