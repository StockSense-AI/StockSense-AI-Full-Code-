import tkinter as tk
import _tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib  # Assuming your model is saved using joblib
import yfinance as yf  # For fetching stock data (if required)

# Load your trained model
# You'll be loading in the directory in which you have StockPrediction.py saved
model = joblib.load('/data/path/to/StockPrediction.py')

def fetch_stock_data(company_name):
    try:
        # Fetch historical stock data using Yahoo Finance
        stock_data = yf.Ticker(company_name)
        hist = stock_data.history(period="1y")  # Last year's data
        # Add your preprocessing logic here to prepare data for the model
        return hist
    except Exception as e:
        messagebox.showerror("Error", f"Could not fetch data for {company_name}: {e}")
        return None

def predict_stock_price():
    company_name = company_entry.get()
    if not company_name:
        messagebox.showerror("Input Error", "Please enter a company name.")
        return

    # Fetch and preprocess data
    data = fetch_stock_data(company_name)
    if data is None:
        return

    # Example: Assume you preprocess `data` to create a feature vector `X`
    X = preprocess_data(data)  # Replace with your preprocessing function
    
    try:
        # Predict stock price
        predicted_price = model.predict(X)
        # Show the predicted price
        messagebox.showinfo("Prediction", f"Predicted stock price for {company_name}: ${predicted_price[0]:.2f}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")

def preprocess_data(data):
    # Add your preprocessing logic here
    # This is a placeholder
    processed_data = data.iloc[-1].values.reshape(1, -1)  # Example: Last row of data
    return processed_data

# Create the GUI window
root = tk.Tk()
root.title("Stock Price Predictor")

# Add GUI elements
tk.Label(root, text="Enter Company Name (e.g., AAPL):").pack(pady=10)
company_entry = tk.Entry(root, width=30)
company_entry.pack(pady=10)

predict_button = tk.Button(root, text="Predict Stock Price", command=predict_stock_price)
predict_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()
