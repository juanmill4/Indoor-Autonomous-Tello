import sys
import traceback
import tellopy
import av
import cv2
import numpy
import time
import cv2.aruco as aruco
import numpy as np
import cv2 as cv
import math

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

def Detected_ArUco_markers(img):
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
            'roll': roll_deg
        }

    return ArUco_marker_orientations

n = 0
SIZE = 94.5

with np.load('/home/user/tello-ai/calib_data/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

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
            img = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            
            Detected_ArUco_markerss = Detected_ArUco_markers(img)
            ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markerss, mtx, dist)
            print(ArUco_marker_orientations)

            for aruco_id, orientation in ArUco_marker_orientations.items():
                corners = Detected_ArUco_markerss[aruco_id]
                center = np.mean(corners[0], axis=0)
                cv2.putText(img, f"ID: {aruco_id}, Yaw: {orientation['yaw']:.2f}, Pitch: {orientation['pitch']:.2f}, Roll: {orientation['roll']:.2f}", 
                            (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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


