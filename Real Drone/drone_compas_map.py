import sys
import traceback
import tellopy
import av
import cv2
import numpy as np
import cv2.aruco as aruco
import math
import time
import random

SIZE = 94.5  # Asegúrate de ajustar este tamaño al de tus marcadores reales

# Variables globales para guardar los datos del giroscopio y la dirección actual
gyro_data = None
current_direction = 0.0  # Dirección inicial apuntando al norte (0 grados)
last_time = None  # Variable para guardar el tiempo del último frame

def handler(event, sender, data, **args):
    global gyro_data
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)
    if event is drone.EVENT_LOG_DATA:
        print(data)
        # Acceder a los datos del giroscopio si data tiene los atributos necesarios
        if hasattr(data, 'imu') and hasattr(data.imu, 'gyro_z'):
            gyro_data = data.imu.gyro_z

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
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, camera_matrix, dist_coeffs)
        rmat = cv2.Rodrigues(rvec)[0]
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
        
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        pitch_deg = (pitch_deg + 180) % 360 - 180
        yaw_deg = (yaw_deg + 180) % 360 - 180
        roll_deg = (roll_deg + 180) % 360 - 180
        
        ArUco_marker_orientations[aruco_id] = {
            'yaw': yaw_deg,
            'pitch': pitch_deg,
            'roll': roll_deg,
            'rvec': rvec[0],
            'tvec': tvec[0]
        }

    return ArUco_marker_orientations

# Función para proyectar puntos 3D a 2D
def project_points(points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return projected_points

def draw_panel(img, rvec, tvec, camera_matrix, dist_coeffs, roll_deg):
    # Definir los vértices del panel aplanado
    panel_points = np.float32([
        [0, 0, 0], [0, SIZE, 0], [SIZE, SIZE, 0], [SIZE, 0, 0],  # Base del panel
        [0, 0, -SIZE/10], [0, SIZE, -SIZE/10], [SIZE, SIZE, -SIZE/10], [SIZE, 0, -SIZE/10]  # Top del panel
    ])

    # Proyectar los puntos 3D a 2D
    img_points = project_points(panel_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = np.int32(img_points).reshape(-1, 2)

    # Dibujar las líneas del panel
    for start, end in zip([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]):
        cv2.line(img, tuple(img_points[start]), tuple(img_points[end]), (0, 255, 0), 2)

    # Dibujar la letra "A" en el centro del panel y rotarla según el ángulo de roll
    A_center = (img_points[0] + img_points[2]) // 2
    A_center = tuple(A_center)
    M = cv2.getRotationMatrix2D(A_center, roll_deg, 1)
    panel_with_A = np.zeros_like(img)
    cv2.putText(panel_with_A, 'A', (A_center[0] - 10, A_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    rotated_panel = cv2.warpAffine(panel_with_A, M, (img.shape[1], img.shape[0]))
    mask = rotated_panel > 0
    img[mask] = rotated_panel[mask]

    # Calcular la distancia y mostrarla en el panel
    distance = np.linalg.norm(tvec)
    cv2.putText(img, f'Distance: {distance:.2f}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Dibujar un mapa simple con la cámara en el centro
    map_center = (500, 100)
    minimap_radius = 70
    cv2.circle(img, map_center, minimap_radius, (255, 0, 0), 2)  # Representa la cámara

    # Escalar la distancia en z para que quepa en el minimapa
    max_distance = 1000  # Define la distancia máxima que quieres representar en el minimapa
    scaled_distance = int((tvec[0][2] / max_distance) * minimap_radius)

    # Asegurar que el marcador no salga del minimapa
    if scaled_distance > minimap_radius:
        scaled_distance = minimap_radius

    # Dibujar el marcador en el minimapa
    marker_position = (map_center[0], map_center[1] - scaled_distance)
    cv2.circle(img, marker_position, 5, (0, 255, 0), -1)

    # Mostrar la distancia en el minimapa
    distance_text = f'{distance:.2f} cm'
    cv2.putText(img, distance_text, (marker_position[0] + 10, marker_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_cube_and_minimap(img, markers, camera_matrix, dist_coeffs):
    if len(markers) < 2:
        return

    ids = list(markers.keys())
    rvec1, tvec1 = markers[ids[0]]['rvec'], markers[ids[0]]['tvec']
    rvec2, tvec2 = markers[ids[1]]['rvec'], markers[ids[1]]['tvec']

    # Calcular la distancia entre los dos marcadores
    distance = np.linalg.norm(tvec1 - tvec2)

    # Definir los vértices del cubo con el centro en la cámara
    half_distance = distance / 2
    cube_points = np.float32([
        [-half_distance, -half_distance, -half_distance], [-half_distance, half_distance, -half_distance],
        [half_distance, half_distance, -half_distance], [half_distance, -half_distance, -half_distance],
        [-half_distance, -half_distance, half_distance], [-half_distance, half_distance, half_distance],
        [half_distance, half_distance, half_distance], [half_distance, -half_distance, half_distance]
    ])

    # Proyectar los puntos 3D del cubo en el espacio de imagen
    img_points = project_points(cube_points, rvec1, tvec1, camera_matrix, dist_coeffs)
    img_points = np.int32(img_points).reshape(-1, 2)

    # Dibujar las líneas del cubo en la imagen
    for start, end in zip([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7, 0, 1, 2, 3]):
        cv2.line(img, tuple(img_points[start]), tuple(img_points[end]), (255, 0, 0), 2)

    # Dibujar los límites del cubo en el minimapa
    map_center = (500, 100)
    minimap_radius = 70
    cv2.circle(img, map_center, minimap_radius, (255, 0, 0), 2)  # Representa la cámara

    for point in cube_points:
        scaled_point = (map_center[0] + int(point[0]), map_center[1] - int(point[2]))
        cv2.circle(img, scaled_point, 5, (0, 255, 0), -1)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

with np.load('/home/user/tello-ai/calib_data/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

angles_list = list(range(-60, 70, 10))
next_angle_time = time.time() + 5
random_angle = None
counter = 5

drone = tellopy.Tello()

try:
    drone.connect()
    drone.wait_for_connection(20.0)
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    drone.subscribe(drone.EVENT_LOG_DATA, handler)

    retry = 3
    container = None
    while container is None and 0 < retry:
        retry -= 1
        try:
            container = av.open(drone.get_video_stream())
        except av.AVError as ave:
            print(ave)
            print('retry...')

    frame_skip = 300

    while True:
        for frame in container.decode(video=0):
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.time()
            img = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            
            Detected_ArUco_markerss = detect_ArUco(img)
            ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markerss, mtx, dist)
            print(ArUco_marker_orientations)

            for aruco_id, orientation in ArUco_marker_orientations.items():
                corners = Detected_ArUco_markerss[aruco_id]
                center = np.mean(corners[0], axis=0)
                cv2.putText(img, f"ID: {aruco_id}, Yaw: {orientation['yaw']:.2f}, Pitch: {orientation['pitch']:.2f}, Roll: {orientation['roll']:.2f}", 
                            (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crear un nuevo frame para el panel 3D
                panel_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_panel(panel_frame, orientation['rvec'], orientation['tvec'], mtx, dist, orientation['roll'])
                cv2.imshow('Panel 3D', panel_frame)
                # Dibujar el cubo y el minimapa
                draw_cube_and_minimap(panel_frame, ArUco_marker_orientations, mtx, dist)
                cv2.imshow('Panel 3D con Cubo', panel_frame)


            # Actualizar la dirección actual basada en los datos del giroscopio
            if gyro_data is not None:
                if last_time is not None:
                    delta_time = start_time - last_time
                    current_direction += gyro_data * delta_time * 180 / np.pi  # Convertir radianes a grados
                    current_direction %= 360  # Mantener la dirección entre 0 y 360 grados
                last_time = start_time

                # Dibujar la brújula en la imagen
                height, width, _ = img.shape
                compass_center = (width - 100, height - 100)
                compass_radius = 50
                cv2.circle(img, compass_center, compass_radius, (0, 255, 0), 2)
                angle_radians = np.deg2rad(current_direction)
                arrow_length = 40
                arrow_end = (int(compass_center[0] + arrow_length * np.sin(angle_radians)),
                             int(compass_center[1] - arrow_length * np.cos(angle_radians)))
                cv2.line(img, compass_center, arrow_end, (0, 0, 255), 2)
                cv2.putText(img, f'Direction: {current_direction:.2f}', (compass_center[0] - 70, compass_center[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Mostrar datos del giroscopio en la imagen
            if gyro_data is not None:
                cv2.putText(img, f'Gyro Z: {gyro_data:.2f}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Original', img)
            if cv2.waitKey(1) == ord("q"):
                break

            if frame.time_base < 1.0/60:
                time_base = 1.0/60
            else:
                time_base = frame.time_base
            frame_skip = int((time.time() - start_time)/time_base)

except Exception as ex:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print(ex)
finally:
    drone.quit()
    cv2.destroyAllWindows()
    cap.release()
