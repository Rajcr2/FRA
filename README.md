# Financial Research Assistant

## Introduction

So, In this project, I have developed a scalable time-series forecasting pipeline using Prophet, designed for quantitative financial forecasting to predict the future stock price of a given company or entity, specifically focusing on its performance month/year ahead. The pipeline integrates critical features such as seasonality, holiday effects, and trend analysis to predict future stock price movements and to provide accurate and actionable forecasts.

### Key Features

1. **Prophet-based time series forecasting**
2. **Incorporates MACD, Signal Line, RSI (7) as regressors**

### Objectives

The primary goal of this project is to create a time-series forecasting system that can:

1. Analyze Real-time historical data to predict future trends.
2. Provide Investor a realistic Forecast.
3. Calculates Stability and Volatility scores Based on these metrics generates insights that will provide investor a assistance.

### Prerequisites
To run this project, you need to install the following libraries:
### Required Libraries

- **Python 3.12+**
- **Pandas**: This library performs data manipulation and analysis also provides powerful data structures like dataframes.
- **Prophet**: A forecasting tool for time-series data, designed to handle trends, seasonality and holidat effects.
- **YFinance**: Yfinance library allows you to easily download historical market data from Yahoo Finance, including stock prices, financials, and more.
- **Fast-Api**: It is Python web framework for building APIs.
- **Streamlit**: Streamlit is a framework that builds interactive, data-driven web applications directly in python.  

Other Utility Libraries : **uvicorn**, **matplotlib**, **numpy**.

### Installation

   ```
   pip install pandas
   pip install prophet
   pip install numpy
   pip install yfinance
   pip install streamlit
   pip install uvicorn
   pip install fastapi
   pip install matplotlib
   ```

### Procedure

1.   Create new directory **'Financial Research Assistant'**.
2.   Inside that directory/folder create new environment.
   
   ```
   python -m venv fra
   ```

  Now, activate this **'fra'** venv.
  
4.   Clone this Repository :

   ```
   git clone https://github.com/Rajcr2/FRA.git
   ```
5.   Now, Install all mentioned required libraries in your environment.
6.   Start Backend server first now with command.
   ```
   uvicorn main:app --reload
   ```
7.   After, that Run **'app.py'** file from Terminal. To activate the dashboard on your browser.
   ```
   streamlit run app.py
   ``` 
7.   Now, move to your browser.
8.   Enter name of a company which stock you want to buy after that it will provide you company stock symbol.
9.   Copy that particular stock symbol & set the model parameters such as changepoint or forecast period and finally click on **'Analyze Stock'** button.
10.  Then within few minutes prophet will train and analyze the model after that you will see forecast results with graphs along with insights.

### Output

![FRA OP](https://github.com/user-attachments/assets/29ebde3a-5ab4-48ce-8295-0be707f9d196)


### Conclusion

Based on error metrics, the model has provided valuable insights. For example, it has identified **TCS** as relatively stable and you can consider for long-term investment, even though Indian market currently facing downturn & vice-versa identified **Adani Green** as highly volatile and mostly suitable for short-term investing/trading. Additionally, the model has effectively captured trends, achieving approximately 65-80% accuracy in actual forecasts.





https://github.com/user-attachments/assets/f1b8fc18-59da-4f5c-8945-38ec5b1a5456









