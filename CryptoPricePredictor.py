import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping 


CRYPTOCURRENCIES = {
    '1': ('btc', 'XXBTZUSD'),  
    '2': ('eth', 'XETHZUSD'),  
    '3': ('xrp', 'XXRPZUSD'),  
    '4': ('ltc', 'XLTCZUSD'),  
}

PREDICT_STEPS = 20  

def get_historical_data(symbol, interval):
    url = f'https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval}'
    response = requests.get(url)
    
 
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.text}')
    
    data = response.json()
    
    if 'result' in data and symbol in data['result']:
     
        prices = [float(candle[4]) for candle in data['result'][symbol]]  
        return prices  
    else:
        raise ValueError("Не удалось получить данные о ценах.")

def select_crypto():
    print("Выберите криптовалюту:")
    for key, (value, _) in CRYPTOCURRENCIES.items():
        print(f"{key}. {value}")
    choice = input("Введите номер криптовалюты: ")
    
    return CRYPTOCURRENCIES.get(choice, CRYPTOCURRENCIES['1'])[1]  

# Функция для выбора интервала
def select_interval():
    print("Выберите интервал графика:")
    print("1. 15 минут")
    print("2. 1 час")
    print("3. 1 день")  
    choice = input("Введите номер интервала (1-3): ")
    
    if choice == '1':
        return 15  # 15 минут
    elif choice == '2':
        return 60  # 1 час
    elif choice == '3':
        return 1440  # 1 день (1440 минут)
    else:
        print("Некорректный выбор, используем 1-часовой интервал по умолчанию.")
        return 60


crypto_symbol = select_crypto()

interval = select_interval()

try:
    prices = get_historical_data(crypto_symbol, interval)
except Exception as e:
    print(f"Ошибка: {e}")
    prices = []

if not prices:
    print("Нет данных для обучения модели.")
else:
    df = pd.DataFrame(prices, columns=['Price'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X_train = []
    y_train = []
    for i in range(PREDICT_STEPS, len(scaled_data)):
        X_train.append(scaled_data[i-PREDICT_STEPS:i])
        y_train.append(scaled_data[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))  
    model.add(LSTM(units=50)) 
    model.add(Dense(units=1))  

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

    model.fit(X_train, y_train, epochs=100, batch_size=16, callbacks=[early_stopping])  

    predictions = []
    last_data = scaled_data[- PREDICT_STEPS:].reshape(1, PREDICT_STEPS, 1)

    for _ in range(PREDICT_STEPS):
        predicted_price = model.predict(last_data)
      
        noise = np.random.normal(0, 0.01)  
        predictions.append(predicted_price[0, 0] + noise) 
 
        last_data = np.append(last_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    predicted_price = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    plt.figure(figsize=(10, 4))  
    plt.plot(df['Price'], color='blue', label='Реальная цена')  
    plt.plot(range(len(df), len(df) + PREDICT_STEPS), predicted_price, color='red', label='Предсказанная цена')  
    plt.title(f'Прогноз цены {crypto_symbol} на {PREDICT_STEPS} шагов вперед ({interval} минут)')
    plt.xlabel('Дни')
    plt.ylabel('Цена в USD ')
    plt.legend()
    plt.show()