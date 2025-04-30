import streamlit as st
import pandas as pd
import requests
from yahooquery import search
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set Streamlit page configuration
st.set_page_config(page_title="Financial Research Assistant", layout="wide")

# Section 2: Stock Analysis & Forecasting
st.title("Financial Research Assistant")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS, RELIANCE.NS)")

# Section 1: Yahoo Query to Search for Ticker Symbol
st.sidebar.title("Find Stock Symbol")
company_name = st.sidebar.text_input("Enter Company Name", "")

if st.sidebar.button("Search Symbol"):
    if company_name.strip():
        results = search(company_name)
        if "names" in results and results["names"]:
            st.sidebar.write("### Matching Stocks:")
        else:
            st.sidebar.warning("No results found. Try a different name.")
    else:
        st.sidebar.warning("Please enter a company name.")

# FastAPI Backend URLs
backend_url = "http://127.0.0.1:8000/analyze/"

# Sidebar: Changepoint Prior Scale and Forecast Periods
changepoint_scale = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.5, 0.22, 0.01)

# Dynamic Tooltip Messages based on Changepoint Scale
if changepoint_scale <= 0.05:
    st.sidebar.info("Stable Forecast: Limited trend flexibility, best for smooth time series.")
elif 0.06 <= changepoint_scale <= 0.15:
    st.sidebar.warning("Moderate Trend Sensitivity: Captures some recent fluctuations.")
elif 0.16 <= changepoint_scale <= 0.25:
    st.sidebar.success("Adaptive to Latest Trends: Balances stability & recent changes.")
else:
    st.sidebar.error("Highly Sensitive: Best for rapid market changes but may overfit.")

forecast_periods = st.sidebar.slider("Forecast Periods (days)", 1, 365, 90)
log_to_mlflow = st.sidebar.checkbox("Log Model to MLflow")  # ✅ Checkbox for MLflow
st.sidebar.write("by Raj Jangam.")

# Fetch Stock Data Button
if st.button("Analyze Stock"):
    with st.spinner("Analyzing stock data..."):
        response = requests.get(
            f"{backend_url}{stock_symbol}",
            params={"forecast_periods": forecast_periods, "changepoint_scale": changepoint_scale, "log_to_mlflow": log_to_mlflow}                    
        )

        if response.status_code == 200:
            result = response.json()
            stock_data = pd.DataFrame(result["data"])
            #forecast_df = pd.DataFrame(result["forecast"])
            forecast_df = pd.DataFrame(result.get("forecast"))

            metrics = pd.DataFrame(result["performance_metrics"])
            accuracy = result["accuracy_results"]
            invest_insights = result["investment_insights"]

            if log_to_mlflow:
                st.success("Model and forecast logged to MLflow ✅")

            # Display Stock Data
            #st.write("### Stock Data Preview", stock_data.tail())      # Earlier Code changed due to Docker Build Issue.
            st.markdown("### Stock Data Preview")  # Title
            st.dataframe(stock_data.tail(), width=500)  # Table with custom width

            # Convert 'ds' to datetime
            stock_data['ds'] = pd.to_datetime(stock_data['ds'])

            # Display Forecast Data
            #st.write("### Forecasted Stock Prices", forecast_df.tail())        # Earlier Code changed due to Docker Build Issue.
            st.markdown("### Forecasted Stock Prices")  # Title
            st.dataframe(stock_data.tail(), width=500)  # Table with custom width

            # Convert 'ds' to datetime for plotting
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

            # Define the past and future range split (70% future, 30% past)
            start_date = forecast_df["ds"].iloc[int(len(forecast_df) * 0.3)]  # 30% past
            end_date = forecast_df["ds"].max()  # Future limit

            # Plot Forecast Data using Matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Predicted", color="blue")
            ax.fill_between(
                forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                color="blue", alpha=0.2, label="Confidence Interval"
            )

            # Formatting Date Axis
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title("Stock Price Forecast")
            ax.legend()

            # Proper Date Formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust date intervals
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format dates correctly
            plt.xticks(rotation=30, ha="right")  # Rotate x-axis labels properly

            # **SET X-AXIS RANGE TO SHOW 30% PAST, 70% FUTURE**
            ax.set_xlim([start_date, end_date])

            st.pyplot(fig)

            ### **📈 Streamlit Interactive Chart (Only 'yhat' vs 'ds')**
            st.write("### Interactive Stock Price Forecast")
            st.line_chart(forecast_df.set_index("ds")[["yhat","yhat_lower"]])

            st.write("### Stability Assessment Summary")
            st.write(accuracy)

            # Display Investment Insights
            st.write("### Investment & Trading Insights")
            for key, value in invest_insights .items():
                st.write(f"**{key}:** {value}")
        
        else:
            st.error(f"Error in API Request: {response.text}")




