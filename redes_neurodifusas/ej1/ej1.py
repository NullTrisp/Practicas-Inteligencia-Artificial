# Importar librerías
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importar los datos 
dataframe = pd.read_csv(r"comprar_alquilar.csv") 
# Mostrar las 10 primeras filas 
dataframe.head(10) 
# ingresos y gastos (comunes, pago_coche,gastos_otros) son mensuales (de 1 personas o 2 si están 
casados) 
# trabajo: 0-sin trabajo 1-autonomo 2-asalariado 3-empresario 4-Autonomos 5-Asalariados 6-
Autonomo y Asalariado 7-Empresario y Autonomo 8 Empresarios o empresario y autónomo 
# estado_civil: 0-soltero 1-casado 2-divorciado 
# hijos: Cantidad de hijos menores (que no trabajan) 
# comprar: 0-mejor alquilar 1-Comprar casa