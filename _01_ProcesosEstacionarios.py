# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install statsmodels

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generar un proceso ARMA(2, 2)
np.random.seed(0)
ar = np.array([1, 0.5, -0.2])
ma = np.array([1, 0.7, 0.2])
arma_process = sm.tsa.ArmaProcess(ar, ma)
simulated_data = arma_process.generate_sample(nsample=100)

# Graficar el proceso
plt.figure(figsize=(12, 4))
plt.plot(simulated_data, marker='o', linestyle='-')
plt.title('Proceso ARMA(2,2)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

# Estimar el modelo ARMA a partir de los datos simulados
modelo_estimado = sm.tsa.ARMA(simulated_data, order=(2, 2))
resultados = modelo_estimado.fit()

# Mostrar resumen de los resultados
print(resultados.summary())
