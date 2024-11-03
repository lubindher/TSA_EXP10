# EXP NO: 10 - IMPLEMENTATION OF SARIMA MODEL FOR NVIDIA STOCK PREDICTION

### Name: Lubindher S
### Register No: 212223240056
### Date: 

## AIM:
To implement SARIMA model using python for Nvidia stock prediction.

## ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
   
## PROGRAM:

### Importing the Packages:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

### Load the data:
```py
data = pd.read_csv('/content/NVIDIA_stock.csv', parse_dates=['Date'], index_col='Date')

plt.figure(figsize=(10, 5))
plt.plot(data)
plt.title('Time Series Data')
plt.show()
```

### Transformation and autocorrelation plotting:
```py
def check_stationarity(ts):
    result = adfuller(ts)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

check_stationarity(data['Volume'])

plot_acf(data['Volume'])
plot_pacf(data['Volume'])
plt.show()
```

### Model creation and prediction:
```py
p = 1  
d = 1  
q = 1  
P = 1  
D = 1  
Q = 1  
s = 12

model = SARIMAX(data['Volume'], order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

predictions = results.get_forecast(steps=12)  
predicted_mean = predictions.predicted_mean
conf_int = predictions.conf_int()

plt.figure(figsize=(10, 5))
plt.plot(data['Volume'], label='Observed')
plt.plot(predicted_mean, label='Predicted', color='red')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink')
plt.title('SARIMA Predictions')
plt.legend()
plt.show()

rmse = np.sqrt(((predicted_mean - data['Volume'].iloc[-12:]) ** 2).mean())
print('Root Mean Squared Error:', rmse)
```

## OUTPUT:

### Original Data:
![image](https://github.com/user-attachments/assets/845b4e0b-9782-4322-8493-742b2eccdc0a)

### ACF and PACF Representation:
![image](https://github.com/user-attachments/assets/5c94e628-758e-4806-ad97-ce7cfe0f88cd)
![image](https://github.com/user-attachments/assets/da027cc8-e68e-43cc-b47d-5e6276724cdb)

### SARIMA Prediction Representation:
![image](https://github.com/user-attachments/assets/6324ad51-98f0-42a6-8aa8-0df580a5c7b5)

## RESULT:
Thus the program run successfully based on the SARIMA model for NVIDIA stock prediction.
