import warnings
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.losses
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from .models import StockData
warnings.filterwarnings("ignore")

def plot_to_base64(fig):
    """Converts a Matplotlib figure to a base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_and_process_data(symbol, threshold=0.6):
    """Loads stock data from the database and processes it while keeping the 'date' column."""

    # Fetch data from the database
    queryset = StockData.objects.filter(symbol=symbol).order_by("date")
    if not queryset.exists():
        return None

    # Convert queryset to DataFrame
    df = pd.DataFrame(list(queryset.values("date", "open_price", "high_price", "low_price", "close_price", "volume")))

    # Convert 'date' column to DateTime format
    df["date"] = pd.to_datetime(df["date"])

    # Rename 'close_price' to 'close' for easier access
    df.rename(columns={"close_price": "close"}, inplace=True)

    # Correlation filter: keep only strongly correlated numerical features (except 'date')
    numerical_features = df.select_dtypes(include=np.number)
    cormap = numerical_features.corr()
    top_correlated_values = cormap["close"][abs(cormap["close"]) > threshold]

    # Ensure 'date' is kept in the final DataFrame
    selected_columns = ["date"] + list(top_correlated_values.index)
    df = df[selected_columns]

    return df

def scale_and_split_data(df):
    """Scales data and splits it into training and testing sets."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_close'] = scaler.fit_transform(df[['close']])

    X, y = [], []
    sequence_length = 10
    for i in range(len(df) - sequence_length):
        X.append(df['scaled_close'].values[i:i + sequence_length])
        y.append(df['scaled_close'].values[i + sequence_length])

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM and CNN
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into 80% training and 20% testing
    split = int(len(X) * 0.8)
    return (X[:split], X[split:], y[:split], y[split:]), scaler

def save_model(model, symbol, model_type):
    """Saves the trained model."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f"{model_dir}/{symbol}_{model_type}.h5" if model_type != "linear_regression" else f"{model_dir}/{symbol}_linear_regression.pkl"
    
    if model_type == "linear_regression":
        joblib.dump(model, model_path)
    else:
        model.save(model_path)

def load_saved_model(symbol, model_type):
    """Loads a saved model if available."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = f"{model_dir}/{symbol}_{model_type}.h5" if model_type != "linear_regression" else f"{model_dir}/{symbol}_linear_regression.pkl"
    
    if os.path.exists(model_path):
        if model_type == "linear_regression":
            return joblib.load(model_path)
        else:
            return load_model(model_path, compile=False)  # ✅ Prevents the 'mse' error
    
    return None

def train_linear_regression(X_train, X_test, y_train, y_test, symbol):
    model = LinearRegression()
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    save_model(model, symbol, "linear_regression")
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    accuracy = 100 - (mape(y_test, y_pred) * 100)
    return model, y_pred, accuracy

def train_ann(X_train, X_test, y_train, y_test, symbol):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    save_model(model, symbol, "ann")
    y_pred = model.predict(X_test).flatten()
    accuracy = 100 - (mape(y_test, y_pred) * 100)
    return model, y_pred, accuracy

def train_lstm(X_train, X_test, y_train, y_test, symbol):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    save_model(model, symbol, "lstm")
    y_pred = model.predict(X_test).flatten()
    accuracy = 100 - (mape(y_test, y_pred) * 100)
    return model, y_pred, accuracy

def train_cnn(X_train, X_test, y_train, y_test, symbol):
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    save_model(model, symbol, "cnn")
    y_pred = model.predict(X_test).flatten()
    accuracy = 100 - (mape(y_test, y_pred) * 100)
    return model, y_pred, accuracy

def predict_future_prices(model, df, last_sequence, scaler, future_days=5):
    """Predicts future stock prices and returns both actual and predicted prices."""
    future_preds = []
    current_input = last_sequence.reshape(1, -1, 1)  # Ensure proper input shape for LSTM/CNN

    for _ in range(future_days):
        # Linear Regression does NOT need `verbose`
        if isinstance(model, LinearRegression):
            next_pred = model.predict(current_input.reshape(1, -1))[0]
        else:
            next_pred = model.predict(current_input, verbose=0)[0, 0]

        future_preds.append(next_pred)
        current_input = np.roll(current_input, -1, axis=1)  # Shift sequence
        current_input[0, -1, 0] = next_pred  # Update last value in sequence

    # Convert predicted prices back to actual scale
    predicted_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten().tolist()

    # Extract actual closing prices for comparison
    actual_prices = df["close"].iloc[-future_days:].tolist()

    return {"actual_prices": actual_prices, "predicted_prices": predicted_prices}



def plot_results(y_test, y_pred, scaler, model_name):
    """Generates actual vs predicted plot and converts it to base64."""
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test_actual, label='Actual', color='#9055FD')
    ax.plot(y_pred_actual, label='Predicted', linestyle="dashed", color='#5FFD53')
    ax.set_title(f'Actual vs Predicted ({model_name})')
    ax.legend()
    ax.grid(True)

    return plot_to_base64(fig)


def plot_existing_predictions(df, future_preds, model_type, future_days):
    """Generates an actual vs predicted prices plot and converts it to base64."""
    if "date" not in df.columns:
        raise ValueError("Date column is missing in DataFrame!")

    df["date"] = pd.to_datetime(df["date"])  # Ensure datetime format

    # Get the last 'future_days' actual prices and their corresponding dates
    actual_dates = df["date"].iloc[-future_days:].tolist()
    actual_prices = df["close"].iloc[-future_days:].tolist()  # Assuming "close" is the actual price column

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot actual stock prices
    ax.plot(actual_dates, actual_prices, marker='o', linestyle="-", color='#4285F4', label="Actual Prices")

    # Annotate actual prices on the plot
    for i, (date, actual_price) in enumerate(zip(actual_dates, actual_prices)):
        ax.annotate(f'{actual_price:.2f}', 
                    (date, actual_price),
                    textcoords="offset points", 
                    xytext=(0, 5), 
                    ha='center', 
                    fontsize=8, 
                    color='#4285F4')

    # Plot predicted prices on the same actual dates
    ax.plot(actual_dates, future_preds, marker='x', linestyle="dashed", color='#9055FD', label="Predicted Prices")

    # Annotate predicted prices on the plot
    for i, (date, pred_price) in enumerate(zip(actual_dates, future_preds)):
        ax.annotate(f'{pred_price:.2f}', 
                    (date, pred_price),
                    textcoords="offset points", 
                    xytext=(0, 5), 
                    ha='center', 
                    fontsize=8, 
                    color='#9055FD')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    # Set proper y-axis step intervals
    min_price = min(min(actual_prices), min(future_preds))
    max_price = max(max(actual_prices), max(future_preds))
    step = max(1, round((max_price - min_price) / 10))  # Dynamic step, at least 1
    ax.set_yticks(np.arange(round(min_price), round(max_price) + 1, step))  # Ensures proper integer steps

    # Labels & title
    ax.set_title(f'Actual vs Predicted Prices ({model_type.upper()})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    return plot_to_base64(fig)

def fill_missing_dates(start_date, future_preds):
    """Generates future dates while skipping weekends."""
    date_range = pd.date_range(start=start_date, periods=len(future_preds) * 2, freq='B')[:len(future_preds)]
    return date_range.tolist(), future_preds

def plot_future_predictions(df, future_preds, model_type):
    """Plots predicted future prices with missing dates handled and annotates predicted prices."""
    if "date" not in df.columns:
        raise ValueError("Date column is missing in DataFrame!")

    df["date"] = pd.to_datetime(df["date"])
    start_date = df["date"].iloc[-1]  # Start from the last available date

    actual_dates, future_preds = fill_missing_dates(start_date, future_preds)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_dates, future_preds, marker='x', linestyle="dashed", color='#5FFD53', label="Predicted Future Prices")
    
    # Annotate the predicted prices on each tick
    for i, txt in enumerate(future_preds):
        ax.annotate(f"{txt:.2f}", (actual_dates[i], future_preds[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='#5FFD53')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    ax.set_title(f'Predicted Future Prices ({model_type.upper()})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    return plot_to_base64(fig)