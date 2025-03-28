import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PREDICT_STEPS = 20
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRYPTOCURRENCIES = {
    '1': ('btc', 'XXBTZUSD'),
    '2': ('eth', 'XETHZUSD'),
    '3': ('xrp', 'XXRPZUSD'),
    '4': ('ltc', 'XLTCZUSD'),
}

def select_crypto():
    print("Выберите криптовалюту:")
    for key, (name, _) in CRYPTOCURRENCIES.items():
        print(f"{key}. {name}")
    choice = input("Введите номер криптовалюты: ")
    return CRYPTOCURRENCIES.get(choice, CRYPTOCURRENCIES['1'])[1]

def select_interval():
    print("Выберите интервал:\n1. 15 минут\n2. 1 час\n3. 1 день")
    choice = input("Введите номер интервала: ")
    return {'1': 15, '2': 60, '3': 1440}.get(choice, 60)

def select_model():
    print("Выберите модель:\n1. LSTM\n2. Transformer")
    choice = input("Введите номер модели: ")
    return 'lstm' if choice == '1' else 'transformer'

def get_historical_data(symbol, interval):
    url = f'https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval}'
    response = requests.get(url).json()
    return [float(candle[4]) for candle in response['result'].get(symbol, [])]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

def train_model(df, model_type):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X_train, y_train = [], []
    for i in range(PREDICT_STEPS, len(scaled_data)):
        X_train.append(scaled_data[i - PREDICT_STEPS:i])
        y_train.append(scaled_data[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    if model_type == 'lstm':
        model = LSTMModel(input_size=X_train.shape[2], hidden_layer_size=64).to(DEVICE)
    else:
        model = TransformerModel(input_size=X_train.shape[2]).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Эпоха {epoch+1}/{EPOCHS}, Потери: {total_loss/len(train_loader):.6f}")

    return model, scaler, scaled_data

def predict(model, scaler, scaled_data):
    model.eval()
    last_data = torch.tensor(scaled_data[-PREDICT_STEPS:].reshape(1, PREDICT_STEPS, 1), dtype=torch.float32, device=DEVICE)
    
    predictions = []
    for _ in range(PREDICT_STEPS):
        with torch.no_grad():
            predicted_price = model(last_data).item()
            predictions.append(predicted_price)
            last_data = torch.cat([last_data[:, 1:, :], torch.tensor([[[predicted_price]]], dtype=torch.float32, device=DEVICE)], dim=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

def plot_results(df, lstm_predictions, transformer_predictions):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Price'], label='Реальная цена', color='blue')

    future_index = np.arange(len(df), len(df) + PREDICT_STEPS)
    plt.plot(future_index, lstm_predictions, label='LSTM Предсказание', color='red', linestyle='dashed')
    plt.plot(future_index, transformer_predictions, label='Transformer Предсказание', color='green', linestyle='dashed')

    plt.title('Прогноз цены с LSTM и Transformer')
    plt.xlabel('Время')
    plt.ylabel('Цена (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

crypto_symbol = select_crypto()
interval = select_interval()

prices = get_historical_data(crypto_symbol, interval)
if not prices:
    print("Нет данных для обучения.")
else:
    df = pd.DataFrame(prices, columns=['Price'])

    lstm_model, scaler, scaled_data = train_model(df, 'lstm')
    transformer_model, _, _ = train_model(df, 'transformer')

    lstm_predictions = predict(lstm_model, scaler, scaled_data)
    transformer_predictions = predict(transformer_model, scaler, scaled_data)

    plot_results(df, lstm_predictions, transformer_predictions)
