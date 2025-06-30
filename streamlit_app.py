import streamlit as st
import requests
import json
import time

FLASK_API_URL = "http://127.0.0.1:5000/predict"

def call_fraud_api(text_message: str = None, transaction_details: dict = None):
    payload = {}
    if text_message:
        payload['textMessage'] = text_message
    
    cleaned_transaction_details = {k: v for k, v in transaction_details.items() if v is not None and v != '' and v != 0.0}
    if cleaned_transaction_details:
        payload.update(cleaned_transaction_details)

    try:
        response = requests.post(FLASK_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "API request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Error connecting to the fraud detection service: {e}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON response from API."}

def process_request(text_msg, tx_details):
    prompt_parts = []
    if text_msg:
        prompt_parts.append(f"Text: '{text_msg}'")
    
    cleaned_tx_details_display = {k: v for k, v in tx_details.items() if v is not None and v != '' and v != 0.0}
    if cleaned_tx_details_display:
        tx_display_parts = []
        for k, v in cleaned_tx_details_display.items():
            tx_display_parts.append(f"{k}={v}")
        if tx_display_parts:
            prompt_parts.append(f"Transaction: {{{'; '.join(tx_display_parts)}}}")

    display_prompt = "; ".join(prompt_parts) if prompt_parts else "Analysis request submitted (no specific data for display)."
    if not display_prompt:
        display_prompt = "Analysis request submitted."

    st.session_state.messages.append({"role": "user", "content": display_prompt})
    with st.chat_message("user"):
        st.markdown(display_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            api_response = call_fraud_api(text_message=text_msg, transaction_details=tx_details)
            
            if "error" in api_response:
                st.error(api_response["error"])
                assistant_response = f"An error occurred: {api_response['error']}"
            else:
                is_fraud = api_response.get("isFraud")
                confidence = api_response.get("confidence")
                model_used = api_response.get("modelUsed")

                if is_fraud:
                    outcome_text = f"**ALERT: This is likely FRAUDULENT.**"
                else:
                    outcome_text = f"**This appears LEGITIMATE.**"
                
                assistant_response = f"""
                {outcome_text}
                
                **Confidence:** {confidence:.2f}%
                **Model Used:** {model_used}
                
                Please let me know if you have more messages or transactions to check.
                """
            
            message_placeholder = st.empty()
            full_response = ""
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})


st.set_page_config(page_title="Fraud Detection Chatbot", page_icon="🔍", layout="wide")
st.title("Fraud Detection Chatbot")
st.markdown(
    """
    Welcome to the Fraud Detection Chatbot. This tool analyzes potential fraud
    using advanced models. Please select an analysis tab below to submit your
    data for a comprehensive assessment.
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello. Please use the tabs below to submit your data."})

with st.container(height=300, border=True):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

tab1, tab2, tab3 = st.tabs(["Hybrid Analysis", "Text Analysis Only", "Transaction Analysis Only"])

with tab1:
    st.subheader("Hybrid Analysis (Text + Transaction)")
    st.write("Provide both a text message and transaction details for a combined assessment.")
    
    hybrid_text_message = st.text_area("Text Message:", height=100, key="text_input_hybrid_tab")
    
    st.markdown("---")
    st.write("**Transaction Details** (use `0` for unknown values)")
    col1_h, col2_h = st.columns(2)
    with col1_h:
        hybrid_amount = st.number_input("Amount (e.g., 10000.0):", min_value=0.0, format="%.2f", key="amount_input_hybrid_tab")
        hybrid_type = st.selectbox("Transaction Type:", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'DEPOSIT', ''], index=5, key="type_input_hybrid_tab")
        hybrid_step = st.number_input("Step (time unit, e.g., 10):", min_value=0, value=0, key="step_input_hybrid_tab")
    with col2_h:
        hybrid_oldbalanceOrg = st.number_input("Original Balance (Sender):", min_value=0.0, format="%.2f", key="oldbalorg_input_hybrid_tab")
        hybrid_newbalanceOrig = st.number_input("New Balance (Sender):", min_value=0.0, format="%.2f", key="newbalorg_input_hybrid_tab")
        hybrid_oldbalanceDest = st.number_input("Original Balance (Receiver):", min_value=0.0, format="%.2f", key="oldbaldest_input_hybrid_tab")
        hybrid_newbalanceDest = st.number_input("New Balance (Receiver):", min_value=0.0, format="%.2f", key="newbaldest_input_hybrid_tab")

    transaction_details_hybrid = {
        'amount': hybrid_amount, 'type': hybrid_type, 'step': hybrid_step,
        'oldbalanceOrg': hybrid_oldbalanceOrg, 'newbalanceOrig': hybrid_newbalanceOrig,
        'oldbalanceDest': hybrid_oldbalanceDest, 'newbalanceDest': hybrid_newbalanceDest
    }
    
    if st.button("Analyze Hybrid", key="analyze_button_hybrid"):
        if not hybrid_text_message and not any(v for k, v in transaction_details_hybrid.items() if v is not None and v != '' and v != 0.0 and k != 'type'):
            st.error("Please provide either a text message or transaction details for Hybrid Analysis.")
        else:
            process_request(hybrid_text_message, transaction_details_hybrid)


with tab2:
    st.subheader("Text Analysis Only")
    st.write("Enter a text message to be analyzed by the FinBERT model.")
    
    text_message_single = st.text_area("Text Message:", height=150, key="text_input_single_tab")
    
    if st.button("Analyze Text", key="analyze_button_text"):
        if not text_message_single:
            st.error("Please provide a text message for analysis.")
        else:
            process_request(text_message_single, {})

with tab3:
    st.subheader("Transaction Analysis Only")
    st.write("Enter financial transaction details for analysis by the tabular model. Use `0` for unknown values.")
    
    col1_t, col2_t = st.columns(2)
    with col1_t:
        transaction_amount = st.number_input("Amount (e.g., 10000.0):", min_value=0.0, format="%.2f", key="amount_input_tab")
        transaction_type = st.selectbox("Transaction Type:", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT', 'DEPOSIT', ''], index=5, key="type_input_tab")
        transaction_step = st.number_input("Step (time unit, e.g., 10):", min_value=0, value=0, key="step_input_tab")
    with col2_t:
        transaction_oldbalanceOrg = st.number_input("Original Balance (Sender):", min_value=0.0, format="%.2f", key="oldbalorg_input_tab")
        transaction_newbalanceOrig = st.number_input("New Balance (Sender):", min_value=0.0, format="%.2f", key="newbalorg_input_tab")
        transaction_oldbalanceDest = st.number_input("Original Balance (Receiver):", min_value=0.0, format="%.2f", key="oldbaldest_input_tab")
        transaction_newbalanceDest = st.number_input("New Balance (Receiver):", min_value=0.0, format="%.2f", key="newbaldest_input_tab")

    transaction_details_single = {
        'amount': transaction_amount, 'type': transaction_type, 'step': transaction_step,
        'oldbalanceOrg': transaction_oldbalanceOrg, 'newbalanceOrig': transaction_newbalanceOrig,
        'oldbalanceDest': transaction_oldbalanceDest, 'newbalanceDest': transaction_newbalanceDest
    }

    if st.button("Analyze Transaction", key="analyze_button_transaction"):
        if not any(v for k, v in transaction_details_single.items() if v is not None and v != '' and v != 0.0 and k != 'type'):
            st.error("Please provide transaction details for analysis.")
        else:
            process_request("", transaction_details_single)
