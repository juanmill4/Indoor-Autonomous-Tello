import cv2
import numpy as np

# Configuración de DPI para la conversión a píxeles
dpi = 300

# Conversión de mm a píxeles
def mm_to_pixels(mm):
    return int(mm * dpi / 25.4)

# Tamaño de la hoja A4 en milímetros convertido a píxeles
a4_width_px = mm_to_pixels(210)
a4_height_px = mm_to_pixels(297)

# Tamaño de cada cuadrado del tablero en milímetros (parametrizable)
square_size_mm = 20  # Por ejemplo, 20 mm

# Tamaño de cada cuadrado en píxeles
square_size_px = mm_to_pixels(square_size_mm)

# Calcular el número de cuadrados en cada dirección
num_squares_width = a4_width_px // square_size_px
num_squares_height = a4_height_px // square_size_px

# Crear imagen en blanco para hoja A4
a4_image = np.ones((a4_height_px, a4_width_px, 3), dtype=np.uint8) * 255

# Dibujar el tablero de ajedrez
for i in range(num_squares_height):
    for j in range(num_squares_width):
        if (i + j) % 2 == 0:
            color = (255, 255, 255)  # Blanco
        else:
            color = (0, 0, 0)  # Negro

        top_left = (j * square_size_px, i * square_size_px)
        bottom_right = ((j + 1) * square_size_px, (i + 1) * square_size_px)
        cv2.rectangle(a4_image, top_left, bottom_right, color, -1)

# Guardar y mostrar la imagen
cv2.imwrite("tablero_ajedrez_A4.png", a4_image)
cv2.imshow("Tablero de Ajedrez A4", a4_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
