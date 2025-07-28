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

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

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

def plot_interactive(df, val_df, predictions, interval_name, crypto_name):
    """Создает интерактивный график с помощью Plotly и включает инструменты рисования."""
    import plotly.graph_objects as go
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Историческая цена', line=dict(color='royalblue')))
    
    fig.add_vline(x=val_df.index[0], line=dict(color='gray', dash='dash'))

    colors = {'LSTM': 'red', 'Transformer': 'green'}
    for name, preds in predictions.items():
        fig.add_trace(go.Scatter(x=preds['index'], y=preds['values'], mode='lines', name=f'Прогноз {name}', line=dict(color=colors[name], dash='dash')))

    fig.update_layout(
        title=f'Прогноз цены {crypto_name} (интервал: {interval_name})',
        xaxis_title='Время',
        yaxis_title='Цена (USD)',
        template='plotly_white',
        hovermode='x unified',
        legend_title_text='Легенда'
    )

    # Добавляем аннотацию для начала валидации
    fig.add_annotation(
        x=val_df.index[0],
        y=0.5,
        text='Начало валидации',
        showarrow=False,
        yref='paper'
    )

    # Включаем инструменты рисования
    config = {
        'modeBarButtonsToAdd': [
            'drawline',        # Рисование линий
            'drawopenpath',    # Рисование открытых контуров
            'drawclosedpath',  # Рисование замкнутых контуров
            'drawcircle',      # Рисование кругов
            'drawrect',        # Рисование прямоугольников
            'eraseshape'       # Удаление нарисованных фигур
        ]
    }

    fig.show(config=config)
# --- Основная функция ---

def main():
    crypto_name, crypto_symbol = select_option(CRYPTOCURRENCIES, "Выберите криптовалюту:", '1')
    interval_minutes, interval_name = select_option(INTERVALS, f"Выберите интервал для {crypto_name}:", '2')

    df = get_historical_data(crypto_symbol, interval_minutes)
    if df is None or len(df) < SEQ_LENGTH * 2:
        print("Недостаточно данных для анализа. Попробуйте больший интервал или другую валюту.")
        return

    train_size = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:train_size], df.iloc[train_size:]

    
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_df)
    scaled_val_data = scaler.transform(val_df)
    
    X_train, y_train = create_sequences(scaled_train_data, SEQ_LENGTH)
    X_val, y_val = create_sequences(scaled_val_data, SEQ_LENGTH)

    if len(X_val) == 0:
        print("❌ Ошибка: Недостаточно данных для создания валидационного набора. Попробуйте больший период.")
        return

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models_to_train = {'LSTM': LSTMModel(), 'Transformer': TransformerModel()}
    all_predictions = {}

   
    for name, model_instance in models_to_train.items():
        trained_model = train_model(model_instance, train_loader, val_loader)
        
        last_sequence_scaled = scaler.transform(df.iloc[-SEQ_LENGTH:])
        future_preds = predict_future(trained_model, scaler, last_sequence_scaled, PREDICT_STEPS)
        
        last_time = df.index[-1]
        freq_str = f"{interval_minutes}min"
        future_index = pd.date_range(start=last_time, periods=PREDICT_STEPS + 1, freq=freq_str)[1:]
        
        all_predictions[name] = {'values': future_preds, 'index': future_index}

  
    plot_interactive(df, val_df, all_predictions, interval_name, crypto_name)

if __name__ == "__main__":
    main()
