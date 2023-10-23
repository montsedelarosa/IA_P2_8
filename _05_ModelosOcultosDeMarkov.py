# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install hmmlearn

import numpy as np
from hmmlearn import hmm

# Definir un modelo oculto de Markov
modelo_hmm = hmm.MultinomialHMM(n_components=2)

# Definir las probabilidades iniciales de los estados
modelo_hmm.startprob_ = np.array([0.6, 0.4])

# Definir las matrices de transición de estados
modelo_hmm.transmat_ = np.array([[0.7, 0.3],
                                 [0.4, 0.6]])

# Definir las probabilidades de emisión de observaciones
modelo_hmm.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                    [0.6, 0.3, 0.1]])

# Generar una secuencia de observaciones
observaciones, estados_ocultos = modelo_hmm.sample(10)

# Estimar los estados ocultos a partir de las observaciones
estados_estimados = modelo_hmm.predict(observaciones)

print("Secuencia de Observaciones:", observaciones)
print("Estados Ocultos Estimados:", estados_estimados)
