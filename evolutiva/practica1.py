# Importar librerías a utilizar
import numpy as np
import matplotlib as plt
import random

# lista números aleatorios
def listaAleatorios(n):
    lista = [0] * n
    for i in range(n):
        lista[i] = int(100*random.random())/100
    return lista


# función de fitness
def fitness(x):
    y = np.abs((x-5)/(2+np.sin(x)))
    return y


def emparejamiento(k):
    # Individuo 1
    if lista[k] > prob1:
        apto1 = "NO"
    else:
        apto1 = "SI"
    k = k+1

    # Individuo 2
    if lista[k] > prob1:
        apto2 = "SI"
    else:
        apto2 = "NO"
    k = k+1

    # Probabilidad de emparejamiento
    if lista[k] > prob_empar:
        pempar = "NO"
    else:
        pempar = "SI"

    k = k+1

    print("Aptitud para emparejamiento -> Individuo 1: ",
          apto1, "  Individuo 2: ", apto2)

    if (pempar == "SI") and (apto1 == "SI") and (apto2 == "SI"):
        print("Hay emparejamiento en esta generación")
        # Punto de corte
        if lista[k] < 0.33:
            pcorte = 1
        elif lista[k] > 0.33 and lista[k] < 0.66:
            pcorte = 2
        else:
            pcorte = 3
        print("Punto de corte para emparejamiento: ", pcorte)
        # Nuevos individuos
        aux1 = cromosoma1
        aux2 = cromosoma2
        cromosoma1 = aux1[0:pcorte]+aux2[pcorte:Lcrom]
        cromosoma2 = aux2[0:pcorte]+aux1[pcorte:Lcrom]
        print("Nuevos Individuos tras emparejamiento")
        print("Individuo 1 :", cromosoma1)
        print("Individuo 2 :", cromosoma2)
    else:
        print("No hay emparejamiento en esta generación")

    return


def mutacion(k):
    # Mutación
    # Individuo 1
    j = 0
    for i in range(k, k+Lcrom):
        if lista[i] > prob_mut:
            print("Se produce mutación en el individuo 1 en la posición ", j)
            if cromosoma1[j] == 1:
                cromosoma1[j] = 0
            else:
                cromosoma1[j] = 1
        j = j+1

    k = k+Lcrom

    # Individuo 2
    j = 0
    for i in range(k, k+Lcrom):
        if lista[i] > prob_mut:
            print("Se produce mutación en el individuo 2 en la posición ", j)
            if cromosoma2[j] == 1:
                cromosoma2[j] = 0
            else:
                cromosoma2[j] = 1
        j = j+1

    k = k+Lcrom

    print("Resultado de la mutación")
    print("Individuo 1 :", cromosoma1)
    print("Individuo 2 :", cromosoma2)

    return


def resultadofitness():
    # Fitness de individuos
    x1 = cromosoma1[3]*1+cromosoma1[2]*2+cromosoma1[1]*2*2+cromosoma1[0]*2*2*2
    x2 = cromosoma2[3]*1+cromosoma2[2]*2+cromosoma2[1]*2*2+cromosoma2[0]*2*2*2

    f1 = fitness(x1)
    f2 = fitness(x2)
    print("Fitness -> Individuo 1 (x=", x1, "): ",
          f1, "  Individuo 2 (x=", x2, "): ", f2)

    # Probabilidad de individuos
    prob1 = f1/(f1+f2)
    prob2 = f2/(f1+f2)

    print("Probabilidad -> Individuo 1 (p=", prob1, ")",
          "  Individuo 2 (p=", prob2, ")")

    return f1, f2, x1, x2


def inicializarcromosomas(k):
    # Cromosomas iniciales
    print("GENERACIÓN INICIAL")
    # Individuo 1
    #cromosoma1 = [0]  * Lcrom
    j = 0
    for i in range(k, k+Lcrom):
        if lista[i] > prob_0_1:
            cromosoma1[j] = 1
        else:
            cromosoma1[j] = 0
        j = j+1

    k = k+Lcrom

    print("Individuo 1 :", cromosoma1)

    # Individuo 2
    #cromosoma2 = [0]  * Lcrom
    j = 0
    for i in range(k, k+Lcrom):
        if lista[i] > prob_0_1:
            cromosoma2[j] = 1
        else:
            cromosoma2[j] = 0
        j = j+1

    k = k+Lcrom

    print("Individuo 2 :", cromosoma2)

    return


# Variables
# Longitud del cromosoma: 4
# Conjunto de elementos del cromosoma: {0,1}
# Número de individuos de la población: 2
# Para la creación de la primera generación:
# Probabilidad del elemento '0': número aleatorio < 0.5
# Probabilidad del elemento '1':  número aleatorio > 0.5
# Probabilidad de emparejamiento (crossover): 0.7
# Probabilidad de mutación: 0.3
Lcrom = 4
n_ind = 2

prob_0_1 = 0.5
prob_empar = 0.7
prob_mut = 0.3
pos_aleat = 0
prob1 = 0
prob2 = 0
max_fit = 0
max_gen = 0
max_x = 0
x1 = 0
x2 = 0
f1 = 0
f2 = 0
cromosoma1 = [0] * Lcrom
cromosoma2 = [0] * Lcrom
aux1 = cromosoma1
aux2 = cromosoma2

# Crear la lista aleatoria de números
n_aleat = 1000
lista = listaAleatorios(n_aleat)
print("lista aleatoria", lista)


# Inicializar cromosomas
inicializarcromosomas(pos_aleat)
pos_aleat = pos_aleat+2*Lcrom


# Aplicar evolución a nuevas generaciones
n_gen = 20
for g in range(0, n_gen):
    print("GENERACIÓN ", g)
    f1, f2, x1, x2 = resultadofitness()
    # Emparejamiento
    emparejamiento(pos_aleat)
    pos_aleat = pos_aleat+3
    # Mutación
    mutacion(pos_aleat)
    pos_aleat = pos_aleat+2*Lcrom

    # Máximo fitness
    if (f1 > max_fit):
        max_fit = f1
        max_x = x1
        max_gen = g
    if (f2 > max_fit):
        max_fit = f2
        max_x = x2
        max_gen = g
    print("Máximo de la función: ", max_fit,
          " (x= ", max_x, " Generación: ", max_gen, ")")
