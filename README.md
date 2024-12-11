# **Stock Price Predictor Project**

## **Overview**

This project predicts stock prices using historical data and a machine learning model. It includes a linear regression model trained on stock features and a GUI application for real-time predictions.

---

## **Features**

1. **Stock Data Fetching**\*: Retrieves historical stock data using Yahoo Finance API.\*  
2. **Feature Engineering**\*: Adds moving averages, lagged values, and performs data preprocessing.\*  
3. **Machine Learning Model**\*: Uses Linear Regression for stock price prediction.\*  
4. **Evaluation Metrics**\*: Calculates MAE, RMSE, and R² scores to evaluate model               performance.\*  
5. **Visualization**\*: Plots actual vs. predicted stock prices.\*  
6. **Interactive GUI**\*: A PyQt5-based interface allows users to input a company ticker symbol  and get stock price predictions.\*

---

## **Getting Started**

### **Prerequisites**

1. Python 3.8 or higher  
2. Install the required dependencies:  
   pip install -r requirements.txt

### **Dependencies**

The following libraries are required:

* yfinance  
* pandas  
* scikit-learn  
* matplotlib  
* PyQt5  
* numpy

---

### **Installation**

Clone the repository:  
git clone https://github.com/StockSense-AI/StockSense-AI-Full-Code-

1. cd StockSense-AI-Full-Code-  
2. Install dependencies:  
   pip install -r requirements.txt  
3. Set up the project: Ensure that the `Stock_Predictor.py` and `Stock_PredictorGUI.py` files       are in the same directory.  
4. Launch the GUI:  
   python Stock_PredictorGUI.py  
   * Input a stock ticker (e.g., AAPL) in the provided text field.  
   * Click "Predict Stock Price" to get predictions.

---

## **Code Structure**

### **Files**

* Stock_Predictor.py : Contains the core functionality for data fetching, preprocessing, model training, and evaluation.  
* Stock_PredictorGUI.py : Contains the PyQt5-based GUI for user interaction.

### **Workflow**

1. Fetch stock data using Yahoo Finance API.  
2. Engineer features such as moving averages and lagging indicators, then clean the data.  
3. Split the data into training and testing sets.  
4. Train a linear regression model on the training data.  
5. Evaluate the model using metrics (MAE, RMSE, and R²).  
6. Display the predictions in the GUI and visualize actual vs. predicted prices.

---

## **Visualization**

The project visualizes the model's performance by plotting actual vs. predicted stock prices. The GUI includes a feature to view these results graphically.

---

## **Future Enhancements**

1. Add support for additional machine learning models (e.g., Random Forest, Neural Networks).  
2. Implement advanced preprocessing techniques for improved predictions.  
3. Enhance the GUI with more user-friendly features and detailed visualizations.

---

## **Contributors**

* **Makari Green**  
* **Danny Adebiyi**

Developers and maintainers of the Stock Price Predictor project.

