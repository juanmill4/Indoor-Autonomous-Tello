import sys
import traceback
import tellopy
import av
import cv2  # for avoidance of pylint error
import numpy
import time
import cv2.aruco as aruco
import numpy as np
import cv2 as cv

MIN_TIME_BETWEEN_DIRECTION_CHANGES = 0.5  # tiempo mínimo en segundos entre cambios de dirección
MARKER_LOST_TIMEOUT = 1  # tiempo en segundos antes de considerar el marcador perdido
last_marker_4_seen_time = None
marker_data = {}
# Preparar el diccionario y los parámetros de ArUco
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters_create()

def handler(event, sender, data, **args):
    global prev_flight_data
    global x
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
         print(data)
         time.sleep(5)

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
        rmat = cv.Rodrigues(rvec)[0]
        sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rmat[2,1], rmat[2,2])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            pitch = np.arctan2(-rmat[1,2], rmat[1,1])
            yaw = np.arctan2(-rmat[2,0], sy)
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


def follow_marker(drone, pitch_actual, roll_actual, throttle_actual, marker_detected, tvec, estado, id, marker_data):
    global contador_pitch_cero, roll_contador_cero, contador_throttle_cero, last_marker_seen_time, last_marker_4_seen_time
    global last_valid_pitch, last_valid_roll, last_valid_throttle, yaw_desired, initial_yaw1, yaw_accumulated, yaw_in_range_count, pass_start_time, yaw_target
    global mapping_mode, mapping_completed
    
    pitch = 0
    roll = 0
    throttle = 0
    yaw = 0
    current_time = time.time()
    pi = np.pi
    
    print(estado)

    if 5 in marker_data:
        x = marker_data[5]['tvec'][0]
        y = marker_data[5]['tvec'][1]
        print(y)
        z = marker_data[5]['tvec']
        z = np.linalg.norm(z)


    # Actualizar el tiempo de detección de la etiqueta 4
    if id == 4:
        last_marker_4_seen_time = current_time

    # Gestión del estado de centrado en roll
    if estado == "centrado_roll":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima_roll = -100
            distancia_maxima_roll = 100
            roll_max = 0.2
            roll_min = 0.1

            if roll_contador_cero >= 16:
                print("Roll mantenido en 0 después de 16 veces consecutivas. Iniciando control de throttle.")
                roll_objetivo = 0
                roll_contador_cero = 0
                estado = "centrado_throttle"
            else:
                if x > distancia_maxima_roll:
                    diferencia = x - distancia_maxima_roll
                    roll_objetivo = max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif x < distancia_minima_roll:
                    diferencia = distancia_minima_roll - x
                    roll_objetivo = -max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    roll_objetivo = 0

                if distancia_minima_roll <= x <= distancia_maxima_roll:
                    roll_contador_cero += 1
                    print("Dron centrado en el eje X, roll ajustado a 0.")
                else:
                    roll_contador_cero = 0

            last_marker_seen_time = current_time        
            last_valid_roll = roll_objetivo  # Guardar el último roll válido
        else:
            roll_objetivo = last_valid_roll

        # Suavizar la transición del roll actual al roll objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        roll = roll_actual + (roll_objetivo - roll_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            drone.set_roll(roll)
            print(f"Dron moviéndose con un roll de {roll:.3f}.")

    elif estado == "centrado_throttle":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima_throttle = -200
            distancia_maxima_throttle = -50
            throttle_max = 0.4
            throttle_min = 0.1

            if contador_throttle_cero >= 16:
                print("Throttle mantenido en 0 después de 16 veces consecutivas. Iniciando control de pitch.")
                throttle_objetivo = 0
                contador_throttle_cero = 0
                estado = "control_pitch"
            else:
                if y > distancia_minima_throttle:
                    diferencia = y - distancia_minima_throttle
                    throttle_objetivo = -max(throttle_min, min(throttle_max, throttle_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif y < distancia_maxima_throttle:
                    diferencia = distancia_maxima_throttle - y
                    throttle_objetivo = max(throttle_min, min(throttle_max, throttle_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    throttle_objetivo = 0

                if distancia_minima_throttle <= y <= distancia_maxima_throttle:
                    contador_throttle_cero += 1
                    print("Dron centrado en el eje Y, throttle ajustado a 0.")
                else:
                    contador_throttle_cero = 0

            last_marker_seen_time = current_time        
            last_valid_throttle = throttle_objetivo  # Guardar el último throttle válido
        else:
            throttle_objetivo = last_valid_throttle

        # Suavizar la transición del throttle actual al throttle objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        throttle = throttle_actual + (throttle_objetivo - throttle_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            drone.set_throttle(throttle)
            print(f"Dron moviéndose con un throttle de {throttle:.3f}.")


    # Gestión del estado de control de pitch
    elif estado == "control_pitch":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima = 400
            distancia_maxima = 700
            pitch_max = 0.3  # Reducción del pitch máximo para suavizar el movimiento
            pitch_min = 0.1  # Pitch mínimo

            if contador_pitch_cero >= 16:
                print("Pitch mantenido en 0 después de 16 veces consecutivas.")
                pitch_objetivo = 0
                contador_pitch_cero = 0
                yaw_target = None
                estado = "stop"
            else:
                if z > distancia_maxima:
                    diferencia = z - distancia_maxima
                    pitch_objetivo = max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                elif z < distancia_minima:
                    diferencia = distancia_minima - z
                    pitch_objetivo = -max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                else:
                    pitch_objetivo = 0

                if distancia_minima <= z <= distancia_maxima:
                    contador_pitch_cero += 1
                    print("Dron dentro del rango objetivo, pitch ajustado a 0.")
                else:
                    contador_pitch_cero = 0

            last_marker_seen_time = current_time
            last_valid_pitch = pitch_objetivo  # Guardar el último pitch válido
        else:
            pitch_objetivo = last_valid_pitch  # Mantener el último pitch válido

        # Suavizar la transición del pitch actual al pitch objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        pitch = pitch_actual + (pitch_objetivo - pitch_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            drone.set_pitch(pitch)
            print(f"Dron moviéndose con un pitch de {pitch:.3f}.")



    elif estado == 'mapping':
        drone.set_pitch(0)
        drone.set_roll(0)
        drone.set_throttle(0)
        yaw_d = 0.6  # Ajuste continuo del yaw
        drone.set_yaw(yaw_d)
        print("Marcador no detectado por más de 3 segundos. Ajustando yaw para buscar marcador.")
        if marker_detected and (id & 0b111) == 0b101:
            print('Marcador detectado nuevamente.')
            estado = 'control_pitch'
            drone.set_yaw(0)

    elif estado == 'stop':
        drone.set_yaw(0)
        drone.set_roll(0)
        drone.set_pitch(0)
        drone.set_throttle(0)
        if id == 5:
            estado = "control_pitch_door"
        elif id == 4:
            print('Close Door')

    elif estado == "control_pitch_door":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima = 1500
            distancia_maxima = 1700
            pitch_max = 0.3  # Reducción del pitch máximo para suavizar el movimiento
            pitch_min = 0.1  # Pitch mínimo

            if contador_pitch_cero >= 16:
                print("Pitch mantenido en 0 después de 16 veces consecutivas.")
                pitch_objetivo = 0
                contador_pitch_cero = 0
                estado = "centrado_roll_door"
            else:
                if z > distancia_maxima:
                    diferencia = z - distancia_maxima
                    pitch_objetivo = max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                elif z < distancia_minima:
                    diferencia = distancia_minima - z
                    pitch_objetivo = -max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                else:
                    pitch_objetivo = 0

                if distancia_minima <= z <= distancia_maxima:
                    contador_pitch_cero += 1
                    print("Dron dentro del rango objetivo, pitch ajustado a 0.")
                else:
                    contador_pitch_cero = 0

            last_marker_seen_time = current_time
            last_valid_pitch = pitch_objetivo  # Guardar el último pitch válido
        else:
            pitch_objetivo = last_valid_pitch  # Mantener el último pitch válido

        # Suavizar la transición del pitch actual al pitch objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        pitch = pitch_actual + (pitch_objetivo - pitch_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            drone.set_pitch(pitch)
            print(f"Dron moviéndose con un pitch de {pitch:.3f}.")


    elif estado == "centrado_roll_door":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima_roll = -150
            distancia_maxima_roll = 50
            roll_max = 0.3
            roll_min = 0.1

            if roll_contador_cero >= 16:
                print("Roll mantenido en 0 después de 16 veces consecutivas. Iniciando estado 'pass'.")
                roll_objetivo = 0
                roll_contador_cero = 0
                estado = "stop_pass"
            else:
                if x > distancia_maxima_roll:
                    diferencia = x - distancia_maxima_roll
                    roll_objetivo = -max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif x < distancia_minima_roll:
                    diferencia = distancia_minima_roll - x
                    roll_objetivo = max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    roll_objetivo = 0

                if distancia_minima_roll <= x <= distancia_maxima_roll:
                    roll_contador_cero += 1
                    print("Dron centrado en el eje X, roll ajustado a 0.")
                else:
                    roll_contador_cero = 0

            last_marker_seen_time = current_time        
            last_valid_roll = roll_objetivo  # Guardar el último roll válido
        else:
            roll_objetivo = last_valid_roll

        # Suavizar la transición del roll actual al roll objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        roll = roll_actual + (roll_objetivo - roll_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            drone.set_roll(roll)
            print(f"Dron moviéndose con un roll de {roll:.3f}.")



    elif estado == "stop_pass":
        if marker_detected and (id & 0b111) == 0b101:
            throttle_objetivo = -0.3
    
            last_marker_seen_time = current_time        
            last_valid_throttle = throttle_objetivo  # Guardar el último throttle válido
        else:
            throttle_objetivo = last_valid_throttle
            # Cambiar a estado 'centrado_roll_door' si la etiqueta 5 no es detectada por más de 0.5 segundos
            if (current_time - last_marker_seen_time) >= 0.7:
                print("Etiqueta 5 no detectada por más de 0.7 segundos. Cambiando a estado 'centrado_roll_door'.")
                estado = "pass"
                last_marker_seen_time = 0  # Resetear el tiempo de última detección
    
        # Suavizar la transición del throttle actual al throttle objetivo
        suavizado = 0.15  # Factor de suavizado, ajusta según sea necesario
        throttle = throttle_actual + (throttle_objetivo - throttle_actual) * suavizado
    
        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            drone.set_throttle(throttle)
            print(f"Dron moviéndose con un throttle de {throttle:.3f}.")



    elif estado == 'pass':
        if last_marker_4_seen_time is not None and (current_time - last_marker_4_seen_time) < 0.3:
            print("Etiqueta 4 detectada recientemente. Esperando.")
            drone.set_yaw(0)
            drone.set_roll(0)
            drone.set_pitch(0)
            drone.set_throttle(0)
        else:
            pitch_speed = 0.6  # Velocidad de pitch deseada
        
            if pass_start_time is None:
                pass_start_time = current_time
        
            if current_time - pass_start_time < 6:
                drone.set_pitch(pitch_speed)
                print(f"Manteniendo pitch a {pitch_speed} durante {current_time - pass_start_time:.2f} segundos.")
            else:
                drone.set_pitch(0)
                print("Estado 'pass' completado. Volviendo a estado 'stop'.")
                estado = 'mapping'
                mapping_mode = True
                mapping_completed = False
                pass_start_time = None  # Resetear el tiempo de inicio para la próxima vez

    return pitch, roll, throttle, estado

video = "video_dron.mp4"
fps = 30
resolucion = (1920,1080)
codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para formato MP4
video_writer = cv2.VideoWriter(video, codec, fps, resolucion)

SIZE = 94.5

with np.load('/home/user/tello-ai/calib_data/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

drone = tellopy.Tello()
contador_pitch_cero = 0
roll_contador_cero = 0
contador_throttle_cero = 0
pitch_actual = 0
roll_actual = 0
throttle_actual = 0
last_marker_seen_time = 0
last_valid_pitch = 0  # Inicializar el último pitch válido
last_valid_roll = 0
last_valid_throttle = 0
estado = "mapping"  # Estado inicial
initial_yaw1 = None
yaw_accumulated = 0
yaw_in_range_count = 0
pass_start_time = None
yaw_target = None

try:
    drone.connect()
    drone.wait_for_connection(30.0)
    bate = drone.EVENT_FLIGHT_DATA
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

    retry = 3
    container = None
    while container is None and 0 < retry:
        retry -= 1
        try:
            container = av.open(drone.get_video_stream())
        except av.AVError as ave:
            print(ave)
            print('retry...')

    # skip first 300 frames
    frame_skip = 300
    found_marker = False
    distance_checks_passed = 0  # Contador para las comprobaciones de distancia exitosas
    drone.takeoff()


    while True:
        for frame in container.decode(video=0):
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.time()
            img = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            gray = cv.cvtColor(numpy.array(frame.to_image()), cv.COLOR_BGR2GRAY)
            # Detección de marcadores ArUco
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            marker_detected = len(corners) > 0

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)
                Detected_ArUco_markers = detect_ArUco(img)
                ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markers, mtx, dist)
                for i, aruco_id in enumerate(ids.flatten()):
                    tvec = tvecs[i][0]
                    distance = np.linalg.norm(tvec)
                    angle = np.arctan2(tvec[0], tvec[2])  # Ángulo en radianes utilizando X e Y
                    marker_data[aruco_id] = {'distance': distance, 'angle': angle, 'tvec': tvec}
                    print(marker_data)
                for i, id in enumerate(ids):
                    # Dibujar cuadrado y ID
                    aruco.drawDetectedMarkers(img, corners, ids)
                    pitch_actual, roll_actual, throttle_actual, estado = follow_marker(drone, pitch_actual, roll_actual, throttle_actual, marker_detected, tvec, estado, id, marker_data)
            
            else:
                pitch_actual, roll_actual, throttle_actual, estado = follow_marker(drone, pitch_actual, roll_actual, throttle_actual, marker_detected, [0, 0, 0], estado, id=None, marker_data=marker_data)
            




            #video_writer.write(images_resized)  # Escribe el cuadro en el archivo de video
            cv2.imshow('Original', img)
            cv2.waitKey(1)

            key = cv.waitKey(1)

            if key == ord("q"):
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
    video_writer.release()
    cv2.destroyAllWindows()
