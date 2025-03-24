import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import os
from groq import Groq
from dotenv import load_dotenv
import json

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI config
st.set_page_config(page_title="💳 AI Cost Categorization & Forecast", page_icon="📈", layout="wide")
st.title("💳 AI Cost Categorization & Expense Forecasting")

# File uploader
uploaded_file = st.file_uploader("Upload your bank statement (Excel with 'Date', 'Description', 'Amount')", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Validate input
        required_cols = {'Date', 'Description', 'Amount'}
        if not required_cols.issubset(set(df.columns)):
            st.error("❌ Excel must include 'Date', 'Description', and 'Amount' columns.")
            st.stop()

        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df.dropna(subset=['Date', 'Description', 'Amount'], inplace=True)

        st.subheader("📄 Uploaded Transactions")
        st.dataframe(df)

        # Use AI to categorize expenses
        st.subheader("🧠 AI Cost Categorization in Progress...")

        client = Groq(api_key=GROQ_API_KEY)
        categorization_prompt = f"""
You are an expert accountant. Classify each transaction below into a cost category (e.g., Travel, SaaS, Payroll, Office Supplies, etc).
Return the response as JSON with an added field called 'Category' for each transaction.

Transactions:
{df[['Date', 'Description', 'Amount']].to_json(orient='records')}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert accountant for SaaS companies."},
                {"role": "user", "content": categorization_prompt}
            ],
            model="llama3-8b-8192",
        )

        import json

# Clean the LLM output and try parsing it
raw_ai_response = response.choices[0].message.content.strip()

# Try to extract the JSON from the LLM response (in case it's wrapped in text)
try:
    # Attempt to find the JSON part if LLM added explanation or code block markers
    if "```json" in raw_ai_response:
        raw_ai_response = raw_ai_response.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw_ai_response:
        raw_ai_response = raw_ai_response.split("```")[1].strip()

    parsed_json = json.loads(raw_ai_response)
    categorized_data = pd.DataFrame(parsed_json)

except Exception as e:
    st.error("❌ Failed to parse AI response. Here's the raw output for debugging:")
    st.code(raw_ai_response)
    st.stop()

        st.success("✅ Transactions Categorized")
        st.dataframe(categorized_data)

        # Forecasting using Prophet
        st.subheader("📈 Forecasting Future Expenses")
        category_to_forecast = st.selectbox("Select a Category to Forecast", categorized_data['Category'].unique())

        df_forecast = categorized_data[categorized_data['Category'] == category_to_forecast].copy()
        df_forecast = df_forecast.groupby('Date')['Amount'].sum().reset_index()
        df_forecast.columns = ['ds', 'y']
        df_forecast['y'] = -df_forecast['y']  # Invert sign for expenses

        periods_input = st.slider("Months to Forecast", 1, 12, 3)

        model = Prophet()
        model.fit(df_forecast)

        future = model.make_future_dataframe(periods=periods_input * 30)
        forecast = model.predict(future)

        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)

        # AI Commentary
        st.subheader("🤖 AI Expense Commentary")
        data_for_ai = categorized_data.to_dict(orient='records')
        prompt = f"""
You are a Head of FP&A. Analyze the categorized bank transactions provided below. Your goal is to:

- Summarize major categories of spend.
- Highlight trends in monthly costs.
- Forecast risks based on current cost trajectory.
- Provide actionable insights.

Data:
{json.dumps(data_for_ai)}
        """

        commentary_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a senior FP&A analyst at a SaaS company."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        commentary = commentary_response.choices[0].message.content
        st.write(commentary)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
