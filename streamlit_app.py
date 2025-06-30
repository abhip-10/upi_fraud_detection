import gradio as gr
import requests
import json
import time


FLASK_API_URL = "http://127.0.0.1:5000/predict"

def call_fraud_api(text_message: str = None, transaction_details: dict = None):
    """
    Calls the Flask API for fraud prediction.
    Only sends data relevant to the selected analysis type.
    """
    payload = {}
    if text_message:
        payload['textMessage'] = text_message
    

    cleaned_transaction_details = {
        k: v for k, v in transaction_details.items() 
        if v is not None and (v != '' if isinstance(v, str) else v != 0.0)
    }
    
    if cleaned_transaction_details:
        payload.update(cleaned_transaction_details)

    try:
        response = requests.post(FLASK_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "API request timed out. Please ensure your Flask backend is running."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Error connecting to the fraud detection service: {e}. Please ensure your Flask backend is running."}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON response from API. Invalid response from backend."}

def predict_fraud(analysis_type, text_message, amount, tx_type, step, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    """
    This function will be called by Gradio to make predictions.
    It adapts inputs based on the selected analysis_type.
    """
    transaction_details = {
        'amount': amount,
        'type': tx_type,
        'step': step,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }

    if analysis_type == "Text Analysis Only":
        if not text_message:
            return "Error: Please provide a text message for text analysis."
        api_response = call_fraud_api(text_message=text_message, transaction_details={})
    
    elif analysis_type == "Transaction Analysis Only":
       
        has_tx_details = any(v is not None and (v != '' if isinstance(v, str) else v != 0.0) for k, v in transaction_details.items() if k != 'type') or (tx_type and tx_type != '')
        if not has_tx_details:
            return "Error: Please provide transaction details for transaction analysis."
        api_response = call_fraud_api(text_message=None, transaction_details=transaction_details)
    
    else: 
        has_text = bool(text_message)
        has_tx_details = any(v is not None and (v != '' if isinstance(v, str) else v != 0.0) for k, v in transaction_details.items() if k != 'type') or (tx_type and tx_type != '')

        if not has_text and not has_tx_details:
            return "Error: Please provide either a text message or transaction details for Hybrid Analysis."
        api_response = call_fraud_api(text_message=text_message, transaction_details=transaction_details)

    if "error" in api_response:
    
        return f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #ef9a9a; color: #c62828; font-weight: bold;">
            An error occurred: {api_response['error']}
        </div>
        """
    else:
        is_fraud = api_response.get("isFraud")
        confidence = api_response.get("confidence")
        model_used = api_response.get("modelUsed")

        if is_fraud:
            outcome_text = f"ALERT: This transaction is highly suspicious and likely FRAUDULENT." # No Markdown bolding here
            background_color = "#ffebee" 
            border_color = "#ef9a9a"
            text_color = "#c62828"
        else:
            outcome_text = f"This transaction appears LEGITIMATE." 
            background_color = "#e8f5e9" 
            border_color = "#a5d6a7"
            text_color = "#2e7d32"
        
        return f"""
        <div style="padding: 15px; border-radius: 10px; background-color: {background_color}; border: 1px solid {border_color}; color: {text_color};">
            <h3 style="font-size: 1.2em; margin-bottom: 8px; font-weight: bold;">{outcome_text}</h3>
            <p style="margin-bottom: 4px;"><strong>Confidence:</strong> {confidence:.2f}%</p>
            <p><strong>Model Used:</strong> {model_used}</p>
            <p style="margin-top: 10px; font-size: 0.9em; color: #555;">
                Our analysis provides a probabilistic assessment. For critical decisions, always consult with a financial expert.
            </p>
        </div>
        """


with gr.Blocks(theme=gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#e3f2fd", c100="#bbdefb", c200="#90caf9", c300="#64b5f6", c400="#42a5f5",
        c500="#2196f3", c600="#1e88e5", c700="#1976d2", c800="#1565c0", c900="#0d47a1",
        c950="#0a3a80"
    ),
    secondary_hue=gr.themes.Color(
        c50="#eceff1", c100="#cfd8dc", c200="#b0bec5", c300="#90a4ae", c400="#78909c",
        c500="#607d8b", c600="#546e7a", c700="#455a64", c800="#37474f", c900="#263238",
        c950="#1a2226"
    ),
    neutral_hue=gr.themes.Color(
        c50="#f5f5f5", c100="#eeeeee", c200="#e0e0e0", c300="#bdbdbd", c400="#9e9e9e",
        c500="#757575", c600="#616161", c700="#424242", c800="#212121", c900="#000000",
        c950="#0a0a0a" 
    )
), css="""
    /* Custom CSS for a more professional look */
    .gradio-container {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6; /* Light gray background */
    }
    h1 {
        text-align: center;
        color: #1976d2; /* Primary blue */
        font-size: 2.5em;
        margin-bottom: 0.5em;
        font-weight: bold;
    }
    h2 {
        color: #37474f; /* Secondary dark gray */
        font-size: 1.8em;
        margin-bottom: 1em;
        font-weight: bold;
    }
    h3 {
        color: #455a64; /* Secondary medium gray */
        font-size: 1.4em;
        margin-bottom: 0.8em;
        font-weight: bold;
    }
    p {
        color: #607d8b; /* Secondary light gray */
        font-size: 1em;
        line-height: 1.5;
    }
    /* Style for the tab buttons */
    .gr-tabs-nav button {
        font-weight: 500;
        color: #6b7280;
        transition: all 0.2s ease-in-out;
    }
    .gr-tabs-nav button.selected {
        color: #1976d2 !important; /* Primary blue for active tab */
        border-bottom: 2px solid #1976d2 !important;
        font-weight: 600;
    }
    /* Input fields styling */
    .gr-form-control {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: none !important;
    }
    .gr-form-control:focus-within {
        border-color: #64b5f6 !important; /* Primary blue on focus */
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2) !important;
    }
    /* Button styling */
    .primary-button {
        background-color: #1976d2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out !important;
        width: 100%;
    }
    .primary-button:hover {
        background-color: #1565c0 !important;
        transform: translateY(-1px) !important;
    }
    .primary-button:active {
        transform: translateY(0) !important;
    }
    /* Result box styling (from previous HTML version) */
    #component-output-text { /* Target the specific Gradio Markdown component */
        padding: 20px;
        border-radius: 10px;
        margin-top: 32px;
        text-align: center;
        font-weight: 500;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    #component-output-text h3 {
        font-size: 1.4em;
        margin-bottom: 10px;
        font-weight: bold;
        color: inherit; /* Inherit color from parent div */
    }
    #component-output-text p {
        margin-bottom: 4px;
        font-size: 0.95em;
        color: inherit; /* Inherit color from parent div */
    }
    #component-output-text .disclaimer {
        margin-top: 15px;
        font-size: 0.85em;
        color: #555;
    }
""") as demo:
    gr.Markdown(
        """
        <h1>Analysis Dashboard</h1>
        """
    )

  
    output_text = gr.Markdown(
        """
        <div style="padding: 15px; border-radius: 10px; background-color: #e3f2fd; border: 1px solid #90caf9; color: #1976d2; text-align: center;">
            <p style="font-size: 1.1em; font-weight: bold;">Analysis Result Will Appear Here</p>
            <p style="font-size: 0.9em; color: #42a5f5;">Submit a form above to initiate a fraud detection scan.</p>
        </div>
        """
    )
    
    with gr.Tab("Hybrid Analysis", elem_id="hybrid-tab-container"):
        gr.Markdown("## Hybrid Analysis: Text & Transaction Data")
        gr.Markdown("Provide both a text message and transaction details for a combined, robust assessment.")
        hybrid_text_input = gr.Textbox(label="Text Message for Analysis:", lines=4, placeholder="e.g., 'Urgent: Your account has been compromised. Click here to verify.'", interactive=True)
        
        gr.Markdown("---")
        gr.Markdown("### Transaction Details (Enter `0` or leave empty for unknown values)")
        
        
        hybrid_amount = gr.Number(label="Transaction Amount:", value=0.0, precision=2, interactive=True)
        hybrid_tx_type = gr.Dropdown(
            ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'DEPOSIT'], 
            label="Transaction Type:", 
            value=None,
            allow_custom_value=False,
            interactive=True 
        )
        hybrid_step = gr.Number(label="Time Step (e.g., 10):", value=0, interactive=True)
        hybrid_oldbalanceOrg = gr.Number(label="Sender's Original Balance:", value=0.0, precision=2, interactive=True)
        hybrid_newbalanceOrig = gr.Number(label="Sender's New Balance:", value=0.0, precision=2, interactive=True)
        hybrid_oldbalanceDest = gr.Number(label="Receiver's Original Balance:", value=0.0, precision=2, interactive=True)
        hybrid_newbalanceDest = gr.Number(label="Receiver's New Balance:", value=0.0, precision=2, interactive=True)

        with gr.Row():
            with gr.Column():
                hybrid_amount
                hybrid_step
                hybrid_newbalanceOrig
                hybrid_newbalanceDest
            with gr.Column():
                hybrid_tx_type
                hybrid_oldbalanceOrg
                hybrid_oldbalanceDest

        hybrid_btn = gr.Button("Perform Hybrid Analysis", elem_classes="primary-button")
        hybrid_btn.click(
            fn=predict_fraud,
            inputs=[
                gr.State("Hybrid Analysis"), 
                hybrid_text_input,
                hybrid_amount,
                hybrid_tx_type,
                hybrid_step,
                hybrid_oldbalanceOrg,
                hybrid_newbalanceOrig,
                hybrid_oldbalanceDest,
                hybrid_newbalanceDest
            ],
            outputs=output_text
        )

    with gr.Tab("Text Analysis Only", elem_id="text-tab-container"):
        gr.Markdown("## Text Message Analysis")
        gr.Markdown("Enter a text message to be analyzed for suspicious language patterns by our FinBERT model.")
        text_only_input = gr.Textbox(label="Text Message for Analysis:", lines=6, placeholder="e.g., 'URGENT: Your bank account requires immediate verification. Click this link.'", interactive=True)
        text_only_btn = gr.Button("Analyze Text Message", elem_classes="primary-button")
        text_only_btn.click(
            fn=predict_fraud,
            inputs=[
                gr.State("Text Analysis Only"), 
                text_only_input,
                *([gr.State(0.0)] * 7) 
            ],
            outputs=output_text
        )

    with gr.Tab("Transaction Analysis Only", elem_id="transaction-tab-container"):
        gr.Markdown("## Transaction Details Analysis")
        gr.Markdown("Enter financial transaction details for analysis by our tabular model. Use `0` or leave empty for unknown values.")
        
        
        transaction_amount = gr.Number(label="Transaction Amount:", value=0.0, precision=2, interactive=True)
        transaction_tx_type = gr.Dropdown(
            ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'DEPOSIT'], 
            label="Transaction Type:", 
            value=None, 
            allow_custom_value=False,
            interactive=True
        )
        transaction_step = gr.Number(label="Time Step (e.g., 10):", value=0, interactive=True)
        transaction_oldbalanceOrg = gr.Number(label="Sender's Original Balance:", value=0.0, precision=2, interactive=True)
        transaction_newbalanceOrig = gr.Number(label="Sender's New Balance:", value=0.0, precision=2, interactive=True)
        transaction_oldbalanceDest = gr.Number(label="Receiver's Original Balance:", value=0.0, precision=2, interactive=True)
        transaction_newbalanceDest = gr.Number(label="Receiver's New Balance:", value=0.0, precision=2, interactive=True)

        with gr.Row():
            with gr.Column():
                transaction_amount
                transaction_step
                transaction_newbalanceOrig
                transaction_newbalanceDest
            with gr.Column():
                transaction_tx_type
                transaction_oldbalanceOrg
                transaction_oldbalanceDest


        transaction_only_btn = gr.Button("Analyze Transaction Details", elem_classes="primary-button")
        transaction_only_btn.click(
            fn=predict_fraud,
            inputs=[
                gr.State("Transaction Analysis Only"), 
                gr.State(""), 
                transaction_amount,
                transaction_tx_type,
                transaction_step,
                transaction_oldbalanceOrg,
                transaction_newbalanceOrig,
                transaction_oldbalanceDest,
                transaction_newbalanceDest
            ],
            outputs=output_text
        )

demo.launch(pwa=True)
