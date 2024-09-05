import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

# Checker board size
CHESS_BOARD_DIM = (13, 9)

# The size of Square in the checker board.
SQUARE_SIZE = 36  # millimeters weboots

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


calib_data_path = "../calib_data/droidcam"
CHECK_DIR = os.path.isdir(calib_data_path)


if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')

else:
    print(f'"{calib_data_path}" Directory already Exists.')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(
    -1, 2
)
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Arrays to store object points and image points from all the images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane.

# The images directory path
image_dir_path = "/home/user/tello-ai/tello-yollov7-ia/images"

files = os.listdir(image_dir_path)
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)
    # print(imagePath)

    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret == True:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)


# h, w = image.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)

total_error = 0
total_points = 0  # Inicializa un contador para los puntos totales
errors_per_image = []  # Lista para guardar el error por imagen

for i in range(len(obj_points_3D)):
    imgpoints2, _ = cv.projectPoints(obj_points_3D[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(img_points_2D[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    total_error += error
    total_points += len(imgpoints2)
    errors_per_image.append(error)  # Guarda el error de la imagen actual en la lista

# Calcular el error medio
mean_error = total_error / len(obj_points_3D)
print(f"Error de reproyección total: {total_error}")
print(f"Error de reproyección medio por imagen: {mean_error}")
print(f"Número total de puntos utilizados: {total_points}")

# Graficar los errores por imagen
plt.figure(figsize=(10, 6))
plt.plot(errors_per_image, label='Error de Reproyección por Imagen', marker='o', linestyle='-', color='blue')
plt.axhline(y=0.5, color='r', linestyle='--', label='Umbral Aceptable (0.5 píxeles)')
plt.xlabel('Número de Imagen')
plt.ylabel('Error de Reproyección (en píxeles)')
plt.title('Error de Reproyección por Imagen vs. Umbral Aceptable')
plt.legend()
plt.grid(True)
plt.show()


print("calibrated")

print("duming the data into one files using numpy ")
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("-------------------------------------------")

print("loading data stored using numpy savez function\n \n \n")

data = np.load(f"{calib_data_path}/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("loaded calibration data successfully")
cv.destroyAllWindows()
