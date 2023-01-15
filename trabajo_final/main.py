# %% [markdown]
# # Limitaciones
# - Las sub imagenes deben ser de un tamaño mayor o igual a 100x100 pixeles
# - Las sub imagenes deben de tener un color de fondo distinto al color de la imagen base

# %%
# Importar librerias necesarias
import pytesseract
import cv2
import pyzbar.pyzbar as pyzbar
import os


# %%
# Funcion que encapsula la librería pytesseract que extrae el texto digital o manual a un string
def image_to_string(image, lang="eng") -> str:
    return pytesseract.image_to_string(image, lang=lang)


# %%
# Funcion para escribir archivos
def save_to_file(filename: str, content: str) -> None:
    with open(filename, "w") as f:
        f.write(content)


# %%
# Funcion para almacenar cualquier tipo de imagen
def save_image(filename: str, image) -> None:
    cv2.imwrite(filename, image)


# %%
# Funcion para eliminar todo contenido de una carpeta en particular
def delete_folder_content(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# %%
# Por cada folder de output ejecutar lo siguiente
for folder_path in ["res", "res/faces", "res/images", "res/qr/"]:
    # En base a la ruta eliminar el contenido
    delete_folder_content(folder_path)


# %%
# Constante que define la imagen a tratar
FILENAME = "data/dario.jpg"


# %%
# Lectura de la imagen base
image = cv2.imread(FILENAME)

# Escritura en disco del output de la imagen base
save_image("res/0_image_inicial.jpg", image)


# %%
# Procesar la imagen base a escala de grises
# (Esto permite eliminar información innecesaria)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Escritura en disco del output de la imagen base a escala de grises
save_image("res/1_imagen_escala_gris.jpg", gray)


# %%
# Procesar la imagen a escala de grises con un desenfoque gausiano 
# (Esto permite suavizar la imagen y eliminar información  innecesaria)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Escritura en disco del output de la imagen a escala de grises con desenfoque
save_image("res/2_imagen_desenfocada.jpg", blurred)


# %%
# Aplicamos un flitro para dejar solamente contornos
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Escritura en disco del output de la imagen con el filtro nuevo
save_image("res/4_imagen_tresh.jpg", thresh)


# %%
# Detectar los codigos QR con la libreria pyzbar
codes = pyzbar.decode(blurred)

# Ejecutar lo siguiente por cada codigo encontrado
for code in codes:
    # Obtenemos sus dimensiones
    x, y, w, h = code.rect

    # Con las dimensiones obtenemos el codigo qr y lo guardamos en disco en el output
    save_image("res/qr/{}.jpg".format(codes.index(code)),
               blurred[y:y+h, x:x+w])

    # De la imagen desenfocada eliminamos el codigo qr quitar información innecesaria
    blurred[y:y+h, x:x+w] = 255

# Escritura en disco del output de la imagen a escala de grises con desenfoque sin codigos qr
save_image("res/3_imagen_sin_qrs.jpg", blurred)


# %%
# Detectar cualquier texto presente en la imagen a escala de grises y desenfocada
save_to_file("res/lectura_imagen.txt", image_to_string(blurred))


# %%
# Obtenemos todos los contornos presentes en la imagen
contours, _ = cv2.findContours(
    cv2.Canny(thresh, 50, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# %%
# Cargamos de disco los kernels para detectar caras y ojos
face_cascade = cv2.CascadeClassifier(
    'data/kernels/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'data/kernels/haarcascade_eye.xml')

# %%
# Por cada contorno ejecutar lo siguiente
for i, cnt in enumerate(contours):
    # Obtener las dimensiones del contorno
    x, y, w, h = cv2.boundingRect(cnt)

    # Determinamos las coordenadas del ROI (Region of interest)
    left = x
    top = y
    right = x + w
    bottom = y + h

    # Obtenemos el ROI (Region of interest) a partir de la imagen base
    roi = image[top:bottom, left:right]

    # Flag que nos ayudará al finalizar a determinar si la imagen contiene una cara o no
    face = False
    # Esta es una validación que nos ayuda a descartar contornos demasiado pequeños que no son una imagen
    if w > 200 and h > 200:
        # Detectamos todas las posibles caras
        faces = face_cascade.detectMultiScale(roi)

        # Por cada cara ejecutar lo siguiente
        for (x, y, w, h) in faces:
            # Pintar un rectangulo en sus coordenadas
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Obtenemos el ROI a escala de grises
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Detectamos todas los posibles ojos
            eyes = eye_cascade.detectMultiScale(gray_roi[y:y+h, x:x+w])
            # Por cada par de ojos ejecutar lo siguiente
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi[y:y+h, x:x+w], (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Determinamos que si hay caras presentes 
            face = True
            break

        # Guardamos en disco los ROI determinando si son imagenes o si contienen caras
        save_image("res/faces/roi_{}.jpg".format(i)
                   if face else "res/images/roi_{}.jpg".format(i), roi)



