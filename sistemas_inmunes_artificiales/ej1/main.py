# Importar librerías
import random
from PIL import Image, ImageDraw, ImageFont
from math import sqrt


def all_pairs(size, shuffle=random.shuffle):
    r1 = range(size)
    r2 = range(size)
    if shuffle:
        shuffle(list(r1))
        shuffle(list(r2))
    for i in r1:
        for j in r2:
            yield (i, j)


def reversed_sections(tour):
    for i, j in all_pairs(len(tour)):
        if i != j:
            copy = tour[:]
            if i < j:
                copy[i:j+1] = reversed(tour[i:j+1])
            else:
                copy[i+1:] = reversed(tour[:j])
                copy[:j] = reversed(tour[i+1:])
            if copy != tour:  # ningún punto devuelve el mismo tour
                yield copy


def swapped_cities(tour):
    for i, j in all_pairs(len(tour)):
        if i < j:
            copy = tour[:]
            copy[i], copy[j] = tour[j], tour[i]
            yield copy


def cartesian_matrix(coords):
    matrix = {}
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            dx, dy = x1-x2, y1-y2
            dist = sqrt(dx*dx + dy*dy)
            matrix[i, j] = dist
    return matrix

# Las coordenadas deben estar en pares x,y por línea y separadas por una coma


def read_coords(coord_file):
    coords = []
    with open(coord_file) as f:
        content = f.readlines()
    for line in content:
        x, y = line.strip().split(',')
        coords.append((float(x), float(y)))
    return coords


def tour_length(matrix, tour):
    total = 0
    num_cities = len(tour)
    for i in range(num_cities):
        j = (i+1) % num_cities
        city_i = tour[i]
        city_j = tour[j]
        total += matrix[city_i, city_j]
    return total


def write_tour_to_img(coords, tour, title, img_file):
    padding = 20
    coords = [(x+padding, y+padding) for (x, y) in coords]
    maxx, maxy = 0, 0
    for x, y in coords:
        maxx = max(x, maxx)
        maxy = max(y, maxy)
    maxx += padding
    maxy += padding
    img = Image.new("RGB", (int(maxx), int(maxy)), color=(255, 255, 255))
    font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    num_cities = len(tour)
    for i in range(num_cities):
        j = (i+1) % num_cities
        city_i = tour[i]
        city_j = tour[j]
        x1, y1 = coords[city_i]
        x2, y2 = coords[city_j]
        d.line((int(x1), int(y1), int(x2), int(y2)), fill=(0, 0, 0))
        d.text((int(x1)+7, int(y1)-5), str(i), font=font, fill=(32, 32, 32))

    for x, y in coords:
        x, y = int(x), int(y)
        d.ellipse((x-5, y-5, x+5, y+5),
                  outline=(0, 0, 0), fill=(196, 196, 196))

    d.text((1, 1), title, font=font, fill=(0, 0, 0))

    del d
    img.save(img_file, "PNG")


def init_random_tour(tour_length):
    tour = list(range(tour_length))
    random.shuffle(tour)
    return tour


def run_hillclimb(init_function, move_operator, objective_function, max_iterations):
    from hillclimb import hillclimb_and_restart
    iterations, score, best = hillclimb_and_restart(
        init_function, move_operator, objective_function, max_iterations)
    return iterations, score, best


out_file_name = "ruta_resultado.png"
max_iterations = 10000
verbose = True
move_operator = reversed_sections
arg = "swapped_cities"
# arg="reversed_sections"

# Fichero con las ccordenadas de las ciudades
city_file = "city100.txt"

# Lectura de las coordenadas de las ciudades
coords = read_coords((city_file))

# Inicializar de forma aleatoria la ruta


def init_function(): return init_random_tour(len(coords))


# Calcular la matriz de distancias
matrix = cartesian_matrix(coords)

# Calcular la longitud de las rutas


def objective_function(tour): return -tour_length(matrix, tour)


# ejecutar el algoritmo de Hill Climbing
iterations, score, best = run_hillclimb(
    init_function, move_operator, objective_function, max_iterations)


print("Iteraciones: ", iterations)
print("Puntuación: ", score)
print("Ruta seleccionada entre ciudades:")
print(best)

# Guardar una imagen de la ruta seleccionada
write_tour_to_img(coords, best, '%s: %f %s' %
                  (city_file, score, arg), out_file_name)
