---

# 📈 Bitcoin Price Prediction using Random Forest Regressor 🧠

Welcome to the Bitcoin Price Prediction project! This repository contains code to load, preprocess, and train a machine learning model to predict Bitcoin closing prices. Using historical data, we employ a RandomForestRegressor to make predictions and evaluate the model's performance. Let's dive into the details! 🚀

## 🗂️ Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## 🌟 Introduction
Predicting Bitcoin prices is both a fascinating and challenging task. This project demonstrates how machine learning can be applied to forecast the closing prices of Bitcoin using historical data.

## 📊 Dataset
The dataset used in this project contains historical Bitcoin prices with the following columns:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

## 🛠️ Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Armanx200/Bitcoin_Price_Prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Bitcoin_Price_Prediction
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Usage
1. Ensure your dataset (`BTC-USD.csv`) is in the project directory.
2. Run the script to train the model and make predictions:
   ```sh
   python BTC.py
   ```

## 📈 Results
The model's performance is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE). Below is the accuracy of the model within a threshold of 2%:

**Accuracy: 99.36%**

### 📊 Actual vs Predicted Close Price Plot
![Plot of Actual vs Predicted Close Price](https://github.com/Armanx200/Bitcoin_Price_Prediction/blob/main/Actual_vs_Predicted.png)

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

---

Made with ❤️ by [Arman Kianian](https://github.com/Armanx200)

---
