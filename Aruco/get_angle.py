import numpy as np
import cv2
import cv2.aruco as aruco
import math
import csv
import random
import time

SIZE = 94.5  # Asegúrate de ajustar este tamaño al de tus marcadores reales

def detect_ArUco(img):
    Detected_ArUco_markers = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        for i in range(len(ids)):
            Detected_ArUco_markers[str(ids[i][0])] = corners[i]
    
    return Detected_ArUco_markers

def Calculate_orientation_in_degree(Detected_ArUco_markers, camera_matrix, dist_coeffs):
    ArUco_marker_orientations = {}

    for aruco_id, corners in Detected_ArUco_markers.items():
        # Estimar la pose del marcador
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, camera_matrix, dist_coeffs)
        
        # Convertir el vector de rotación en una matriz de rotación
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Calcular los ángulos de Euler
        sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rmat[2,1], rmat[2,2])
            yaw = math.atan2(-rmat[2,0], sy)
            roll = math.atan2(rmat[1,0], rmat[0,0])
        else:
            pitch = math.atan2(-rmat[1,2], rmat[1,1])
            yaw = math.atan2(-rmat[2,0], sy)
            roll = 0
        
        # Convertir los ángulos de pitch, yaw y roll de radianes a grados
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        # Ajustar los ángulos al rango [-180, 180)
        pitch_deg = (pitch_deg + 180) % 360 - 180
        yaw_deg = (yaw_deg + 180) % 360 - 180
        roll_deg = (roll_deg + 180) % 360 - 180

        # # Compensar el error del salto de -40º
        # if yaw_deg < -40:
        #     yaw_deg += 40
        
        ArUco_marker_orientations[aruco_id] = {
            'yaw': yaw_deg,
            'pitch': pitch_deg,
            'roll': roll_deg
        }

    return ArUco_marker_orientations

# Función para obtener un ángulo aleatorio sin repetición
def get_random_angle(angles_list):
    if len(angles_list) == 0:
        angles_list = list(range(-60, 70, 10))
    angle = random.choice(angles_list)
    angles_list.remove(angle)
    return angle, angles_list

# Configuración de la cámara
url = 'http://192.168.118.57:4747/video'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

# Carga la matriz de la cámara y los coeficientes de distorsión
with np.load('/home/user/tello-ai/calib_data/droidcam/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

angles_list = list(range(-60, 70, 10))
next_angle_time = time.time() + 5
random_angle = None
counter = 5

# Abrir archivo CSV para guardar los datos
with open('angles_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['random_angle', 'aruco_yaw'])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el marco")
                break

            # Detección de marcadores ArUco
            Detected_ArUco_markers = detect_ArUco(frame)
            # Calcular la orientación de cada marcador
            ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markers, mtx, dist)

            # Mostrar los ángulos calculados para cada marcador detectado
            for aruco_id, orientations in ArUco_marker_orientations.items():
                print(f"ArUco ID {aruco_id}: Yaw: {orientations['yaw']:.2f}°, Pitch: {orientations['pitch']:.2f}°, Roll: {orientations['roll']:.2f}°")

            # Actualizar el contador cada segundo
            remaining_time = int(next_angle_time - time.time())
            if remaining_time != counter:
                counter = remaining_time

            # Mostrar el ángulo aleatorio y el contador en la ventana
            if random_angle is not None:
                cv2.putText(frame, f'Random Angle: {random_angle}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Tomando medida en: {counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostrar la imagen resultante
            cv2.imshow('result', frame)
            if cv2.waitKey(1) == ord("q"):
                break

            # Guardar datos en el CSV justo antes de actualizar el ángulo aleatorio y el contador
            if time.time() >= next_angle_time:
                if Detected_ArUco_markers:
                    for aruco_id in Detected_ArUco_markers:
                        aruco_yaw = ArUco_marker_orientations[aruco_id]['yaw']
                        writer.writerow([random_angle, aruco_yaw])
                else:
                    writer.writerow([random_angle, None])

                random_angle, angles_list = get_random_angle(angles_list)
                next_angle_time = time.time() + 8
                counter = 8

    except Exception as ex:
        print(f"Error: {ex}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
