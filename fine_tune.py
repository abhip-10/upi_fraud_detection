
!pip install transformers[torch] datasets accelerate scikit-learn -q

import torch
import pandas as pd
import numpy as np
import random
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import pipeline

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



def generate_fraud_messages(num_messages):
    templates = [
        "Dear customer your {bank} account will be blocked. Update your PAN card immediately by clicking this link: {url}",
        "Your electricity connection will be disconnected tonight at {time}. Please contact officer Mr. {name} {phone} immediately.",
        "Congratulations! You have been selected for a work-from-home job with a salary of {amount} INR. Click to join: {url}",
        "We have detected a suspicious login from a new device. Please share the OTP {otp} to verify your identity.",
        "Your package from {courier} is on hold at our {city} hub. Pay a small fee of {fee} INR to release it for delivery: {url}",
        "You have won the {lottery} lottery of {amount} Lakhs. Contact this WhatsApp number {phone} to claim your prize.",
        "Your {bank} bank account KYC has expired. Please update it using the link to avoid account suspension: {url}",
        "Get a pre-approved loan of {amount} from {company} with no documentation. Apply now: {url}",
        "URGENT: Your {service} subscription has expired. Update your payment details here to continue watching: {url}",
        "A refund of {amount} INR has been initiated. To receive it in your account, accept this UPI request.",
        "Your {card_type} credit card has been blocked due to suspicious activity. To unblock, please verify your details here: {url}",
        "Get {data}GB free data on your {telco} number. Offer valid for today only. Click here to activate: {url}",
        "DEAR SIR/MADAM, YOU HAVE WON {amount} IN A LUCKY DRAW. TO CLAIM, DEPOSIT {fee} RS AS TAX. CALL {phone}.",
        "Your {e_wallet} account needs re-verification. Failure to do so will result in suspension. Verify now: {url}",
        "Your Demat account will be suspended if you don't update your nomination. Last day today. Update here: {url}",
        "Part time job offer: Earn {amount} daily. Just 2 hours of work. No experience needed. Contact us on WhatsApp: {phone}.",
        "Your Amazon account is locked due to too many failed login attempts. To unlock it, please verify your identity at {url}",
        "Scan this QR code to receive {amount} Rs in your bank account. Open your UPI app and scan now!",
        "Important security alert for your {bank} account. Please call our security team on {phone} to confirm recent activity.",
        "You have been charged Rs. {amount} for a purchase you did not make. To cancel this transaction, click here immediately: {url}"
    ]
    
    options = {
        'bank': ['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB', 'Kotak', 'Bank of Baroda'],
        'time': ['9.30pm', '10:00 PM', '8:30pm', 'midnight'],
        'name': ['Sharma', 'Verma', 'Kumar', 'Singh', 'Gupta'],
        'phone': ['9xxxxxxxxx', '8xxxxxxxxx', '7xxxxxxxxx', '6xxxxxxxxx'],
        'amount': ['25,000', '5,00,000', '10,000', '5000', '2,500', '4,999', '1,850'],
        'url': ['fake-login.com/verify', 'secure-update.net/kyc', 'official-portal.org/login', 'bit.ly/scam-link', 'tinyurl.com/fraud-link'],
        'otp': [str(random.randint(100000, 999999)) for _ in range(10)],
        'courier': ['Amazon', 'Flipkart', 'Delhivery', 'Blue Dart', 'DHL'],
        'city': ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'],
        'fee': ['5', '10', '49', '99', '150'],
        'lottery': ['KBC', 'State Govt', 'Indian Oil', 'PM Lucky Draw'],
        'company': ['Bajaj Finserv', 'Indiabulls', 'Capital First', 'LoanFront'],
        'service': ['Netflix', 'Hotstar', 'Amazon Prime', 'SonyLIV', 'Zee5'],
        'card_type': ['ICICI', 'HDFC', 'SBI', 'Axis', 'RBL', 'OneCard'],
        'data': ['50', '25', '100', '75'],
        'telco': ['Jio', 'Airtel', 'Vodafone Idea', 'BSNL'],
        'e_wallet': ['Paytm', 'PhonePe', 'Google Pay', 'Amazon Pay']
    }
    
    messages = set()
    while len(messages) < num_messages:
        template = random.choice(templates)
        placeholders = {k: random.choice(v) for k, v in options.items() if f"{{{k}}}" in template}
        messages.add(template.format(**placeholders))
    return list(messages)

def generate_legit_messages(num_messages):
    templates = [
        "Your OTP for transaction at {merchant} is {otp}. Do not share this with anyone.",
        "Your order #{order_id} from {merchant} has been shipped. It will be delivered by tomorrow.",
        "REMINDER: Your appointment with Dr. {name} is scheduled for tomorrow at {time}.",
        "Transaction alert: A purchase of INR {amount} was made on your {bank} Card ending {card_digits} at {merchant}.",
        "Your Swiggy order is out for delivery. Our delivery partner will reach you in {minutes} minutes.",
        "INR {amount} has been credited to your account via UPI from {phone}.",
        "Your electricity bill of Rs. {amount} for account {account_no} is due on {date}. Pay via the official app.",
        "Welcome to {airline}. Your flight {flight_no} from {city_from} to {city_to} is confirmed for {date}. Your PNR is {pnr}.",
        "Your updated account balance for A/C no. XXXX{card_digits} is INR {balance}.",
        "Thank you for contacting {company} customer support. Your reference number is {ref_id}.",
        "Your Uber is arriving in {minutes} minutes. Your driver is {name}, in a {car_color} {car_model}.",
        "A login from a new device was detected on your {service} account. If this was you, no action is needed.",
        "Salary of INR {amount} has been credited to your account.",
        "Thanks for payment of Rs {amount} for your {telco} recharge.",
        "Your prescription is ready for pickup at {pharmacy}.",
        "Your {bank} account XXXX{card_digits} has been debited for INR {amount} for a recurring SIP payment.",
        "Your {service} monthly payment of {amount} was successful.",
        "Your cab from {cab_service} is confirmed. Vehicle No. {vehicle_no}. Driver contact: {phone}.",
        "Your refund of Rs. {amount} from {merchant} for order #{order_id} has been processed.",
        "Your {utility} bill for {amount} is generated for the month of June. Please pay before {date}."
    ]
    
    options = {
        'merchant': ['Flipkart', 'Amazon', 'Myntra', 'BigBazaar', 'Zomato', 'Swiggy', 'BookMyShow'],
        'otp': [str(random.randint(100000, 999999)) for _ in range(10)],
        'order_id': [f'#{random.randint(100000, 999999)}-{random.randint(100000, 999999)}' for _ in range(5)],
        'name': ['Gupta', 'Ramesh', 'Suresh', 'Priya'],
        'time': ['11:00 AM', '3:30 PM', '5:00 PM', '10:30 AM'],
        'amount': [str(random.randint(100, 50000)) for _ in range(20)],
        'bank': ['HDFC Bank', 'SBI', 'ICICI Bank', 'Axis Bank'],
        'card_digits': [str(random.randint(1000, 9999)) for _ in range(5)],
        'minutes': [str(random.randint(3, 20)) for _ in range(5)],
        'phone': ['9xxxxxxxxx', 'friend@upi', '7xxxxxxxxx'],
        'account_no': [str(random.randint(100000, 999999)) for _ in range(5)],
        'date': ['15-Jul-2024', '01-Aug-2025', '20-Sep-2024'],
        'airline': ['Vistara', 'IndiGo', 'Air India', 'SpiceJet'],
        'flight_no': ['UK810', '6E-2045', 'AI-505'],
        'city_from': ['DEL', 'BOM', 'BLR', 'MAA'],
        'city_to': ['BOM', 'DEL', 'CCU', 'HYD'],
        'pnr': [str(random.randint(1000000000, 9999999999))],
        'balance': [f"{random.randint(1000, 100000)}.{random.randint(0,99)}"],
        'company': ['SBI', 'Airtel', 'Vodafone', 'Urban Company'],
        'ref_id': [f"SR{random.randint(100000, 999999)}", f'CRN{random.randint(10000, 99999)}'],
        'car_color': ['white', 'black', 'silver', 'blue'],
        'car_model': ['Swift Dzire', 'WagonR', 'i20', 'Creta'],
        'service': ['Google', 'Facebook', 'Microsoft', 'Netflix', 'Zerodha'],
        'telco': ['Jio', 'Airtel', 'Vi'],
        'pharmacy': ['Apollo Pharmacy', 'Netmeds', 'Pharmeasy'],
        'cab_service': ['Ola', 'Uber', 'Meru'],
        'vehicle_no': [f'KA01AB{random.randint(1000, 9999)}'],
        'utility': ['Gas', 'Water', 'Broadband']
    }
    
    messages = set()
    while len(messages) < num_messages:
        template = random.choice(templates)
        placeholders = {k: random.choice(v) for k, v in options.items() if f"{{{k}}}" in template}
        messages.add(template.format(**placeholders))
    return list(messages)

fraud_texts = generate_fraud_messages(500)
legit_texts = generate_legit_messages(500)

data = {
    'text': fraud_texts + legit_texts,
    'label': [1] * len(fraud_texts) + [0] * len(legit_texts)
}

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) 

print("Dataset created:")
print(f"Total messages: {len(df)}")
print("\nClass distribution:")
print(df['label'].value_counts())
print("-" * 30)


model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

if '__index_level_0__' in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', '__index_level_0__'])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(['text', '__index_level_0__'])
else:
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(['text'])

tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Calculated Class Weights: {class_weights}")

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4, # Adjusted for a larger dataset
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    save_total_limit=2,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting model training...")
trainer.train()
print("Training finished.")

print("\n--- Model Evaluation Summary ---")
eval_results = trainer.evaluate()
for key, value in eval_results.items():
    print(f"{key.replace('_', ' ').title():<25}: {value:.4f}")



best_model_path = trainer.state.best_model_checkpoint
print(f"\nBest model saved at: {best_model_path}")

# Create a zip archive of the best model directory
shutil.make_archive("finbert_fraud_model", 'zip', best_model_path)
print("\nModel zipped as 'finbert_fraud_model.zip'.")
print("You can now download this file from the Kaggle file explorer on the right-hand side under 'Output'.")

print("\n--- Interactive Fraud Detection ---")
fraud_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

while True:
    try:
        user_message = input("\nEnter a message to classify (or type 'quit' to exit): ")
        if user_message.lower() == 'quit':
            break
        if not user_message:
            continue
        
        results = fraud_classifier(user_message)
        result = results[0]

        # --- Cleaner Output Formatting ---
        print("\n" + "="*80)
        print(" " * 29 + "Inference Result")
        print("="*80)
        print(f"{'Prediction':<12} | {'Confidence':<15} | {'Score':<7} | {'Message'}")
        print("-"*80)

        label = "Fraud" if result['label'] == 'LABEL_1' else "Legitimate"
        score = result['score']

        if score > 0.9:
            confidence = "High"
        elif score > 0.75:
            confidence = "Medium"
        else:
            confidence = "Low"

        print(f"{label:<12} | {confidence:<15} | {score:<7.4f} | {user_message[:60]}...")
        print("="*80)

    except (KeyboardInterrupt, EOFError):
        # Gracefully handle Ctrl+C or end-of-file in non-interactive environments
        print("\nExiting interactive session.")
        break
