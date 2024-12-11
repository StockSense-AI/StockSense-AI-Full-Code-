import sys
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from Stock_Predictor import fetch_stock_data, preprocess_data, split_data, train_model, evaluate_model

# PyQt5 Application Class
class StockPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.y_test = None
        self.y_pred = None

    def initUI(self):
        self.setWindowTitle("Stock Price Predictor")
        self.setGeometry(100, 100, 600, 400)

        # Layout and Widgets
        layout = QVBoxLayout()

        self.label = QLabel("Enter Stock Symbol (e.g., AAPL):")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.stock_input = QLineEdit(self)
        layout.addWidget(self.stock_input)

        self.predict_button = QPushButton("Predict Stock Price", self)
        self.predict_button.clicked.connect(self.predict_stock_price)
        layout.addWidget(self.predict_button)

        self.visualize_button = QPushButton("Visualize Results", self)
        self.visualize_button.clicked.connect(self.visualize_results)
        self.visualize_button.setEnabled(False)
        layout.addWidget(self.visualize_button)

        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        layout.addWidget(self.result_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def predict_stock_price(self):
        stock_symbol = self.stock_input.text()
        if not stock_symbol:
            self.result_display.setText("Error: Please enter a valid stock symbol.")
            return

        start_date = "2020-01-01"
        end_date = datetime.date.today() - datetime.timedelta(days=1)

        try:
            stock_data = fetch_stock_data(stock_symbol, start_date, str(end_date))
            stock_data = preprocess_data(stock_data)
        except Exception as e:
            self.result_display.setText(str(e))
            return

        if stock_data.empty:
            self.result_display.setText("Error: No data available for the specified stock.")
            return

        try:
            X_train, X_test, y_train, y_test = split_data(stock_data)
        except ValueError as ve:
            self.result_display.setText(f"Error during data split: {ve}")
            return

        model = train_model(X_train, y_train)
        y_pred, mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        self.y_test = y_test
        self.y_pred = y_pred

        next_day_features = stock_data.iloc[-1][['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'Lag_1']].values.reshape(1, -1)
        predicted_price = float(model.predict(next_day_features)[0])

        self.visualize_button.setEnabled(True)

        result_text = (
            f"Stock Symbol: {stock_symbol}\n"
            f"Predicted Price for Next Day: ${predicted_price:.2f}\n"
            f"Mean Absolute Error (MAE): {mae:.2f}\n"
            f"Root Mean Squared Error (RMSE): {rmse:.2f}\n"
            f"R² Score: {r2:.2f}\n\n"
            "Click 'Visualize Results' to see Actual vs Predicted Prices."
        )
        self.result_display.setText(result_text)

    def visualize_results(self):
        if self.y_test is not None and self.y_pred is not None:
            plt.figure(figsize=(10, 6))
            plt.xlabel("Days corresponding to the test data.")  # Title for the X-axis
            plt.ylabel("Stock Prices $ (USD)")  # Title for the Y-axis
            plt.plot(self.y_test.values, label='Actual Prices', color='blue')
            plt.plot(self.y_pred, label='Predicted Prices', color='red', linestyle='--')
            plt.legend()
            plt.title('Actual vs Predicted Stock Prices')
            plt.show()

# Main Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = StockPredictorApp()
    main_window.show()
    sys.exit(app.exec_())
