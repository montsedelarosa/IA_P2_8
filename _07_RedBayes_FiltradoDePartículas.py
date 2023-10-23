# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install numpy

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
num_particulas = 100
tiempo = 50
ruido_proceso = 0.1
ruido_medicion = 1.0

# Función de movimiento en línea recta
def movimiento_lineal(x, ruido):
    return x + ruido * np.random.randn()

# Función de medición (observación)
def medicion_real(x, ruido):
    return x + ruido * np.random.randn()

# Inicialización de partículas y pesos
particulas = np.random.uniform(0, 100, num_particulas)
pesos = np.ones(num_particulas) / num_particulas

# Listas para almacenar estimaciones
estimaciones = []

# Ciclo de tiempo
for t in range(tiempo):
    # Actualizar las partículas utilizando el modelo de movimiento
    particulas = movimiento_lineal(particulas, ruido_proceso)
    
    # Simular una medición
    medicion = medicion_real(t, ruido_medicion)
    
    # Actualizar los pesos basados en la similitud con la medición
    likelihood = np.exp(-0.5 * ((particulas - medicion) / ruido_medicion)**2)
    pesos = pesos * likelihood
    pesos /= np.sum(pesos)
    
    # Resamplear partículas
    indices = np.random.choice(np.arange(num_particulas), size=num_particulas, p=pesos)
    particulas = particulas[indices]
    pesos = np.ones(num_particulas) / num_particulas
    
    # Estimar la posición a partir de las partículas
    estimacion = np.mean(particulas)
    estimaciones.append(estimacion)

# Graficar las estimaciones
plt.plot(range(tiempo), estimaciones, label='Estimaciones', marker='o')
plt.title('Filtrado de Partículas')
plt.xlabel('Tiempo')
plt.ylabel('Estimación de Posición')
plt.grid(True)
plt.legend()
plt.show()
