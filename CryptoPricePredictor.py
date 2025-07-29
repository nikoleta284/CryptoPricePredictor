import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta
import time
import os

# --- Константы ---
SEQ_LENGTH = 30      # Длина последовательности для входа в модель
PREDICT_STEPS = 20   # На сколько шагов вперед предсказываем
EPOCHS = 100         # Количество эпох обучения
BATCH_SIZE = 16      # Размер батча
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRYPTOCURRENCIES = {
    '1': ('BTC', 'XXBTZUSD'),
    '2': ('ETH', 'XETHZUSD'),
    '3': ('XRP', 'XXRPZUSD'),
    '4': ('LTC', 'XLTCZUSD'),
    '5': ('ADA', 'ADAUSD'),
    '6': ('DOT', 'DOTUSD'),
    '7': ('SOL', 'SOLUSD'),
    '8': ('LINK', 'LINKUSD'),
    '9': ('UNI', 'UNIUSD'),
    '10': ('AAVE', 'AAVEUSD'),
    '11': ('SNX', 'SNXUSD'),
    '12': ('YFI', 'YFIUSD'),
    '13': ('COMP', 'COMPUSD'),
    '14': ('MKR', 'MKRUSD'),
    '15': ('BAL', 'BALUSD'),
    '16': ('CRV', 'CRVUSD'),
    '17': ('SUSHI', 'SUSHIUSD'),
    '18': ('ALPHA', 'ALPHAUSD'),
    '19': ('KEEP', 'KEEPUSD'),
    '20': ('REN', 'RENUSD'),
}

INTERVALS = {
    '1': (15, '15 минут'),
    '2': (60, '1 час'),
    '3': (1440, '1 день')
}

# --- Вспомогательные функции ---

def select_option(options, prompt, default_key):
    """Функция для выбора опции пользователем."""
    print(prompt)
    for key, value in options.items():
        display_name = value[0] if isinstance(value, tuple) else value[1]
        print(f"{key}. {display_name}")
    choice = input("Введите номер: ")
    return options.get(choice, options[default_key])

def get_historical_data(symbol, interval):
    """Получает исторические данные с API Kraken."""
    try:
        url = f'https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval}'
        response = requests.get(url, timeout=10).json()
        if 'error' in response and response['error']:
            raise ValueError(f"Ошибка API: {response['error']}")
        
        data_key = next(iter(response['result']))
        data = response['result'].get(data_key, [])
        
        if not data:
            raise ValueError("Нет данных для указанной пары.")
            
        timestamps = [int(candle[0]) for candle in data]
        prices = [float(candle[4]) for candle in data]  # Цена закрытия
        df = pd.DataFrame({'Price': prices}, index=pd.to_datetime(timestamps, unit='s'))
        return df
    except Exception as e:
        print(f"❌ Ошибка при получении данных: {e}")
        return None

# --- Модели ---

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- Подготовка данных и обучение ---

def create_sequences(data, seq_length):
    """Создает последовательности (X) и цели (y) из временного ряда."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=10):
    """Обучает модель с валидацией и ранним остановом."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    model_save_path = 'best_model.pth'
    
    print(f"\n🚀 Обучение модели {type(model).__name__} на {DEVICE}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Эпоха {epoch+1}/{epochs}, Потери при обучении: {train_loss:.6f}, Потери при валидации: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⌛ Ранний останов на эпохе {epoch+1}. Лучшая валидационная потеря: {best_loss:.6f}")
                break
    
    model.load_state_dict(torch.load(model_save_path))
    if os.path.exists(model_save_path):
        os.remove(model_save_path) 
    return model

def predict_future(model, scaler, last_sequence, predict_steps):
    """Авторегрессионное предсказание на N шагов вперед."""
    model.eval()
    
    last_sequence_tensor = torch.tensor(last_sequence.reshape(1, -1, 1), dtype=torch.float32).to(DEVICE)
    
    predictions_scaled = []
    with torch.no_grad():
        for _ in range(predict_steps):
            pred = model(last_sequence_tensor)
            predictions_scaled.append(pred.item())
            
            new_pred_tensor = pred.reshape(1, 1, 1)
            last_sequence_tensor = torch.cat([last_sequence_tensor[:, 1:, :], new_pred_tensor], dim=1)
            
    return scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# --- Визуализация ---

def plot_interactive_with_signals(df, val_df, predictions, interval_name, crypto_name, entry_point, entry_time, exit_point, exit_time):
    """Создает интерактивный график с торговыми сигналами."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Историческая цена', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=predictions['index'], y=predictions['values'], mode='lines', name='Прогноз', line=dict(color='red', dash='dash')))
    fig.add_vline(x=val_df.index[0], line=dict(color='gray', dash='dash'))
    
    # Добавление сигналов
    if entry_point and entry_time:
        fig.add_trace(go.Scatter(x=[entry_time], y=[entry_point], mode='markers', name='Вход', marker=dict(color='green', size=10)))
    if exit_point and exit_time:
        fig.add_trace(go.Scatter(x=[exit_time], y=[exit_point], mode='markers', name='Выход', marker=dict(color='orange', size=10)))
    
    fig.update_layout(
        title=f'Прогноз цены {crypto_name} (интервал: {interval_name})',
        xaxis_title='Время',
        yaxis_title='Цена (USD)',
        template='plotly_white',
        hovermode='x unified',
        legend_title_text='Легенда'
    )
    
    config = {
        'modeBarButtonsToAdd': ['drawline', 'drawcircle', 'drawrect', 'eraseshape']
    }
    fig.show(config=config)

# --- Дополнительные функции ---

def calculate_mse(model, val_loader, criterion=nn.MSELoss()):
    """Вычисляет среднеквадратичную ошибку (MSE) на валидационном наборе."""
    model.eval()
    mse = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            mse += criterion(outputs, targets).item()
    return mse / len(val_loader)

def find_best_crypto(cryptocurrencies, interval_minutes):
    """Находит криптовалюту с наименьшим MSE на валидационном наборе."""
    best_crypto = None
    best_mse = float('inf')
    best_model = None
    best_predictions = None
    best_df = None
    best_val_df = None
    best_scaler = None
    
    for key, (name, symbol) in cryptocurrencies.items():
        print(f"\nАнализ {name}...")
        df = get_historical_data(symbol, interval_minutes)
        if df is None or len(df) < SEQ_LENGTH * 2:
            print(f"Недостаточно данных для {name}.")
            continue
        
        # Разделение данных
        train_size = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:train_size], df.iloc[train_size:]
        
        # Нормализация
        scaler = MinMaxScaler()
        scaled_train_data = scaler.fit_transform(train_df)
        scaled_val_data = scaler.transform(val_df)
        
        # Создание последовательностей
        X_train, y_train = create_sequences(scaled_train_data, SEQ_LENGTH)
        X_val, y_val = create_sequences(scaled_val_data, SEQ_LENGTH)
        
        if len(X_val) == 0:
            print(f"Недостаточно данных для валидации {name}.")
            continue
        
        # Подготовка датасетов
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Обучение модели (используем LSTM как пример)
        model = LSTMModel().to(DEVICE)
        trained_model = train_model(model, train_loader, val_loader)
        
        # Оценка MSE
        mse = calculate_mse(trained_model, val_loader)
        print(f"{name} MSE: {mse:.6f}")
        
        if mse < best_mse:
            best_mse = mse
            best_crypto = name
            best_model = trained_model
            best_df = df
            best_val_df = val_df
            best_scaler = scaler
            
            # Прогноз
            last_sequence_scaled = scaler.transform(df.iloc[-SEQ_LENGTH:])
            future_preds = predict_future(trained_model, scaler, last_sequence_scaled, PREDICT_STEPS)
            last_time = df.index[-1]
            freq_str = f"{interval_minutes}min"
            future_index = pd.date_range(start=last_time, periods=PREDICT_STEPS + 1, freq=freq_str)[1:]
            best_predictions = {'values': future_preds, 'index': future_index}
    
    return best_crypto, best_model, best_df, best_val_df, best_predictions, best_mse, best_scaler

def get_trade_signals(predictions, current_price, profit_target=0.05, entry_threshold=0.01):
    """Определяет точки входа и выхода на основе прогноза."""
    entry_point = None
    exit_point = None
    entry_time = None
    exit_time = None
    
    pred_values = predictions['values']
    pred_index = predictions['index']
    
    # Проверка условий входа
    for i in range(len(pred_values) - 1):
        if (pred_values[i + 1] - current_price) / current_price >= entry_threshold:
            entry_point = pred_values[i]
            entry_time = pred_index[i]
            break
    
    # Проверка условий выхода
    if entry_point:
        for i in range(len(pred_values)):
            if pred_values[i] >= entry_point * (1 + profit_target):
                exit_point = pred_values[i]
                exit_time = pred_index[i]
                break
            elif i > 0 and pred_values[i] < pred_values[i - 1]:  # Начало падения
                exit_point = pred_values[i]
                exit_time = pred_index[i]
                break
    
    return entry_point, entry_time, exit_point, exit_time

# --- Основная функция ---

def main():
    interval_minutes, interval_name = select_option(INTERVALS, "Выберите интервал:", '2')
    
    # Поиск лучшей криптовалюты
    best_crypto, best_model, best_df, best_val_df, best_predictions, best_mse, best_scaler = find_best_crypto(CRYPTOCURRENCIES, interval_minutes)
    
    if not best_crypto:
        print("Не удалось найти подходящую криптовалюту для анализа.")
        return
    
    print(f"\n✅ Лучшая криптовалюта: {best_crypto} с MSE: {best_mse:.6f}")
    
    # Определение торговых сигналов
    current_price = best_df['Price'].iloc[-1]
    entry_point, entry_time, exit_point, exit_time = get_trade_signals(best_predictions, current_price)
    
    if entry_point:
        print(f"Точка входа: {entry_point:.2f} USD в {entry_time}")
    else:
        print("Нет подходящей точки входа.")
    
    if exit_point:
        print(f"Точка выхода: {exit_point:.2f} USD в {exit_time}")
    else:
        print("Нет точки выхода или прогноз не достиг цели прибыли.")
    
    # Визуализация
    plot_interactive_with_signals(best_df, best_val_df, best_predictions, interval_name, best_crypto, entry_point, entry_time, exit_point, exit_time)

if __name__ == "__main__":
    main()
