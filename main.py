from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FastAPI is running. Use /analyze/{symbol} for stock insight & forecast."}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Financial Research Assistant model")

# Function: Compute RSI
def compute_RSI(prices, period=7):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Small epsilon to avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function: Fetch Stock Data
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y")
    df.reset_index(inplace=True)

    df = df[['Date', 'Open', 'Close', 'Volume']]

    # Create 'ds' and 'y' columns for Prophet
    df['ds'] = df['Year'] + '-' + df['Month'] + '-' + df['Day']
    df['y'] = df['Close']  # Prophet's target variable

    # Calculate MACD and Signal Line
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Compute RSI
    df['RSI'] = compute_RSI(df['Close']).rolling(window=5, min_periods=1).mean().fillna(method="bfill")

    return df[['ds', 'y', 'MACD', 'RSI', 'Signal_Line']]

# Function: Train Prophet Model
def train_prophet_model(data, forecast_periods, changepoint_scale, log_to_mlflow=False):
    # Define Prophet Model
    model = Prophet(
        seasonality_mode="additive",
        changepoint_prior_scale=changepoint_scale,
        n_changepoints=30,
        weekly_seasonality=False,
        yearly_seasonality=False
    )

    # Adding Monthly & Quarterly Seasonality
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=7)

    # Add Regressors
    model.add_regressor('MACD') 
    model.add_regressor('Signal_Line')
    model.add_regressor('RSI')

    # Fit Model
    model.fit(data)

    # Forecasting Future Data
    future = model.make_future_dataframe(periods=forecast_periods, freq="D")

    for col in ['MACD', 'RSI', 'Signal_Line']:
        future[col] = data[col].reindex(future.index, fill_value=data[col].iloc[-1])

    forecast = model.predict(future)

    # Cross-Validation (Testing on last 90 days)
    df_cv = cross_validation(model, horizon="90 days")
    df_p = performance_metrics(df_cv)

    # Evaluate Forecast Performance
    accuracy_results = evaluate_forecast_performance(df_p["rmse"], df_p["mape"], df_p["mse"])
    
    if log_to_mlflow:
        with mlflow.start_run():
            mlflow.log_param("changepoint_scale", changepoint_scale)
            mlflow.log_param("forecast_periods", forecast_periods)
            mlflow.log_metric("rmse", df_p["rmse"].mean())
            mlflow.log_metric("mape", df_p["mape"].mean())
            mlflow.log_metric("mse", df_p["mse"].mean())
            mlflow.sklearn.log_model(model, "prophet_model")

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_p, accuracy_results

# Function: Evaluate Forecast Accuracy
def evaluate_forecast_performance(rmse_values, mape_values, mse_values):
    
    accuracy_values = [100 - (mape * 100) for mape in mape_values]
    return {
        "Stability (Min) (%)": round(np.min(accuracy_values), 2),
        "Stability (Max) (%)": round(np.max(accuracy_values), 2),
        "Average Stability (%)": round(np.mean(accuracy_values), 2)
    }

# Function: Analyze Stock Prediction Accuracy
def analyze_stock_accuracy(min_acc, max_acc, avg_acc):
       
    accuracy_range = max_acc - min_acc
    insights = {}

    # Determine Stock Stability
    if avg_acc >= 90 and min_acc >= 85:
        insights["Stability"] = "Very High (Stock is stable with minimal risk)"
        insights["Investment Advice"] = "Best for long-term and passive investors."
    elif avg_acc >= 85 and min_acc >= 75 and max_acc >= 95:
        insights["Stability"] = "High (Stock is generally stable but has occasional spikes)"
        insights["Investment Advice"] = "Good for long-term, can also be used for swing trading."
    elif avg_acc >= 80 and min_acc >= 70:
        insights["Stability"] = "Moderate (Predictable but with some fluctuations)"
        insights["Investment Advice"] = "Safe for long-term but requires periodic review."
    elif avg_acc >= 75 and (min_acc >= 65 or max_acc >= 95):
        insights["Stability"] = "Low-Moderate (Stock fluctuates but has growth potential)"
        insights["Investment Advice"] = "Risky for long-term but good for short-term trading."
    else:
        insights["Stability"] = "Low (Highly Unpredictable, risky for long-term holding)"
        insights["Investment Advice"] = "Best suited for short-term, day trading, or options."

    # Volatility Analysis (Considering max, min, and range)
    if accuracy_range < 5 and avg_acc >= 85:
        insights["Volatility"] = "Very Low (Highly stable, predictable stock)"
        insights["Trading Advice"] = "Safe for long-term SIPs and passive investments."
    elif accuracy_range < 10 and avg_acc >= 80:
        insights["Volatility"] = "Low (Mild fluctuations, good for holding)"
        insights["Trading Advice"] = "Best for holding 3+ years with low risk."
    elif accuracy_range < 20 and avg_acc >= 75:
        insights["Volatility"] = "Moderate (Stock moves but within reasonable limits)"
        insights["Trading Advice"] = "Can be used for swing trading and medium-term strategies."
    elif accuracy_range >= 20 and avg_acc >= 70:
        insights["Volatility"] = "High (Frequent price swings, needs active monitoring)"
        insights["Trading Advice"] = "Ideal for short-term traders."
    else:
        insights["Volatility"] = "Very High (Extremely unpredictable, risky)"
        insights["Trading Advice"] = "Best for aggressive traders, avoid long-term holding."

    return insights

@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str, forecast_periods: int, changepoint_scale: float ):
    """Fetches stock data from Yahoo Finance for the given symbol and returns formatted data."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y")

        if df.empty:
            return {"error": "Invalid stock symbol or no data available."}

        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'Close', 'Volume']]
        
        # Ensure 'Date' column is in correct format
        df['Date'] = pd.to_datetime(df['Date'])

        # Convert DataFrame to preferred format
        formatted_data = get_stock_data(symbol)

        # Convert DataFrame to JSON format
        stock_data = formatted_data.to_dict(orient="records")

        # Train Prophet Model
        #forecast, df_p, accuracy_results = train_prophet_model(stock_data, forecast_periods, changepoint_scale)
        forecast, df_p, accuracy_results = train_prophet_model(formatted_data, forecast_periods, changepoint_scale)

        # Extract accuracy values for insights
        min_acc = accuracy_results["Stability (Min) (%)"]
        max_acc = accuracy_results["Stability (Max) (%)"]
        avg_acc = accuracy_results["Average Stability (%)"]

        # 🔹 Compute Trading & Investment Insights
        investment_insights = analyze_stock_accuracy(min_acc, max_acc, avg_acc)

        return {"symbol": symbol, "data": stock_data, "forecast": forecast.to_dict(orient="records"),
                "performance_metrics": df_p.to_dict(orient="records"),
                "accuracy_results": accuracy_results,
                "investment_insights": investment_insights      
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
