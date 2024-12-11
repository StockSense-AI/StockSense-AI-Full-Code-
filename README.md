# Stock Price Predictor Project

## Overview

This project predicts stock prices using historical data and a machine learning model. It includes a linear regression model trained on stock features and a GUI application for real-time predictions.

---

## Features

1. **Stock Data Fetching**: Retrieves historical stock data using Yahoo Finance API.
2. **Feature Engineering**: Adds moving averages, lagged values, and performs data preprocessing.
3. **Machine Learning Model**: Uses Linear Regression for stock price prediction.
4. **Evaluation Metrics**: Calculates MAE, RMSE, and R² scores to evaluate model performance.
5. **Visualization**: Plots actual vs. predicted stock prices.
6. **Interactive GUI**: A Tkinter-based interface allows users to input a company ticker symbol and get stock price predictions.

---

 1. Clone the repository:
   ```bash
   git clone https://github.com/StockSense-AI/Makari_G.git
   cd Makari_G
2. Install dependencies
    pip install -r requirements.txt
3. Ensure the following libraries are installed:
    yfinance
    pandas
    scikit-learn
    matplotlib
    tkinter
    joblib
4. Save your trained model as a .pkl file in the project directory using joblib
    import joblib
    joblib.dump(model, 'linear_regression_model.pkl')
5. Launch the GUI
    python StockPredictionGUI.py
    Input a stock ticker (e.g., AAPL) in the provided text field.
    Click "Predict Stock Price" to get predictions.

Code Structure

StockPrediction.py: Main script for data fetching, preprocessing, model training, and evaluation.
StockPredictionGUI.py: Tkinter-based GUI for user-friendly stock price prediction.
linear_regression_model.pkl: Saved machine learning model for GUI predictions.

Model Workflow

Fetch stock data using yfinance.
Engineer features: Add moving averages, lagging indicators, and drop missing data.
Train-Test split: Split data into training and testing sets.
Train the linear regression model.
Evaluate using MAE, RMSE, and R² scores.
Predict stock prices using a GUI.
Visualization

The project visualizes the model's performance by plotting actual vs. predicted stock prices, making it easier to understand the model's accuracy.

Future Enhancements

Add support for additional machine learning models (e.g., Random Forest, Neural Networks).
Implement advanced preprocessing techniques for improved predictions.
Enhance the GUI with more user-friendly features and detailed visualizations.
Contributors

Makari Green and Adeolu Adebiyi
Developers and maintainers of the Stock Price Predictor project.



