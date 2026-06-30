
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def download_sp500_data(start_date, end_date):
    """Downloads S&P 500 data from Yahoo Finance."""
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    return sp500_data

def feature_engineering(data):
    """Creates features for the model."""
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = np.where(data['Returns'].shift(-1) > 0, 1, 0)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data.dropna(inplace=True)
    return data

def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def walk_forward_validation(data, train_years, test_months):
    """Performs walk-forward validation."""

    # Reserve the last two years for out-of-sample testing
    out_of_sample_start = data.index[-1] - pd.DateOffset(years=2)
    in_sample_data = data[data.index < out_of_sample_start]
    out_of_sample_data = data[data.index >= out_of_sample_start]




    best_hyperparameters = {}


    # Hyperparameter grid for Ridge Logistic Regression
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # Sliding window for training and testing
    start_date = in_sample_data.index[0]
    end_date = in_sample_data.index[-1]


    train_window = pd.DateOffset(years=train_years)
    test_window = pd.DateOffset(months=test_months)



    current_date = start_date



    while current_date + train_window + test_window <= end_date:
        train_end = current_date + train_window
        test_end = train_end + test_window

        train_set = in_sample_data[(in_sample_data.index >= current_date) & (in_sample_data.index < train_end)]
        test_set = in_sample_data[(in_sample_data.index >= train_end) & (in_sample_data.index < test_end)]

        if train_set.empty or test_set.empty:
            current_date += test_window
            continue

        features = ['SMA_10', 'SMA_50', 'RSI']
        X_train = train_set[features]
        y_train = train_set['Target']
        X_test = test_set[features]
        y_test = test_set['Target']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_auc = -1
        best_c = None



        for c in c_values:
            model = LogisticRegression(penalty='l2', C=c, solver='liblinear', random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred)

            if auc > best_auc:
                best_auc = auc
                best_c = c




        if best_c is not None:
            best_hyperparameters[train_end.strftime('%Y-%m-%d')] = {'C': best_c, 'AUC': best_auc}



        current_date += test_window



    return best_hyperparameters, in_sample_data, out_of_sample_data

def main():
    """Main function to run the analysis."""
    start_date = '2010-01-01'
    end_date = '2023-12-31'

    sp500_data = download_sp500_data(start_date, end_date)
    sp500_data = feature_engineering(sp500_data)

    best_hyperparameters, in_sample_data, out_of_sample_data = walk_forward_validation(sp500_data, train_years=2, test_months=1)

    print("Best Hyperparameters Found During Walk-Forward Validation:")
    for date, params in best_hyperparameters.items():
        print(f"End of Train Set: {date}, C: {params['C']}, AUC: {params['AUC']:.4f}")

if __name__ == '__main__':
    main()
