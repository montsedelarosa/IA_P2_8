# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install statsmodels

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Simular datos de una serie temporal
np.random.seed(0)
t = np.arange(0, 100)
serie_temporal = 0.5 * t + 2 * np.random.randn(100)

# Filtrar la serie temporal utilizando un modelo autoregresivo ARIMA
modelo_arima = sm.tsa.ARIMA(serie_temporal, order=(1, 0, 0))
resultados_arima = modelo_arima.fit()

# Realizar predicciones
predicciones = resultados_arima.predict(start=100, end=110)

# Suavizar la serie temporal utilizando un promedio móvil
ventana = 5
serie_suavizada = np.convolve(serie_temporal, np.ones(ventana)/ventana, mode='same')

# Explicación de la serie temporal
residuos = resultados_arima.resid

# Gráficos
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, serie_temporal)
plt.title('Serie Temporal Original')

plt.subplot(2, 2, 2)
plt.plot(t, resultados_arima.fittedvalues)
plt.title('Filtrado (Modelo ARIMA)')

plt.subplot(2, 2, 3)
plt.plot(t, serie_suavizada)
plt.title(f'Suavizado (Promedio Móvil, Ventana = {ventana})')

plt.subplot(2, 2, 4)
plt.plot(t, residuos)
plt.title('Residuos del Modelo ARIMA')

plt.tight_layout()
plt.show()

# Predicciones
print('Predicciones:')
print(predicciones)
