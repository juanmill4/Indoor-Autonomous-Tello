from controller import Robot
from controller import Keyboard
from math import cos, sin
from pid_controller import pid_velocity_fixed_height_controller

import cv2 as cv
import numpy as np
import time
import os

FLYING_ATTITUDE = 1
MIN_TIME_BETWEEN_DIRECTION_CHANGES = 0.5  # tiempo mínimo en segundos entre cambios de dirección
MARKER_LOST_TIMEOUT = 1  # tiempo en segundos antes de considerar el marcador perdido

# Cargar la matriz de la cámara y los coeficientes de distorsión desde un archivo .npz
with np.load('/home/user/tello-ai/calib_data/webots/crazy/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']
# Tamaño de marcador ArUco en milímetros (ajusta según tus marcadores)
SIZE = 400

# Preparar el diccionario y los parámetros de ArUco
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_250)
parameters = cv.aruco.DetectorParameters_create()

# Ajustar valor dentro del rango -1.0 a 1.0
def fix_range(value):
    return max(min(value, 1.0), -1.0)

# Funciones para controlar el drone
def set_throttle(value):
    global height_diff_desired, height_desired
    height_diff_desired = fix_range(value)
    height_desired += height_diff_desired * dt

def set_yaw(value):
    global yaw_desired
    yaw_desired = fix_range(value)

def set_pitch(value):
    global forward_desired
    forward_desired = fix_range(value)

def set_roll(value):
    global sideways_desired
    sideways_desired = fix_range(value)

def land():
    global height_desired, past_time, past_x_global, past_y_global
    height_desired = 0  # Set desired height to 0 to land
    while True:
        dt = robot.getTime() - past_time
        actual_state = {}
        
        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global) / dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global) / dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, 0, 0, 0, height_desired, roll, pitch, yaw_rate, altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

        if altitude <= 0.05:  # Check if the drone is close enough to the ground
            m1_motor.setVelocity(0)
            m2_motor.setVelocity(0)
            m3_motor.setVelocity(0)
            m4_motor.setVelocity(0)
            print("Drone has landed.")
            break
        robot.step(timestep)

def takeoff():
    global height_desired, past_time, past_x_global, past_y_global
    height_desired = FLYING_ATTITUDE  # Set desired height to the flying attitude
    while True:
        dt = robot.getTime() - past_time
        actual_state = {}
        
        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global) / dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global) / dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, 0, 0, 0, height_desired, roll, pitch, yaw_rate, altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

        if altitude >= FLYING_ATTITUDE - 0.05:  # Check if the drone has reached the desired altitude
            print("Drone has taken off.")
            break
        robot.step(timestep)

def follow_marker(center_x, center_y, frame_center_x, frame_center_y, distancia_a_camara, pitch_actual, roll_actual, throttle_actual, marker_detected, tvec, estado):
    global contador_pitch_cero, roll_contador_cero, contador_throttle_cero, last_marker_seen_time, last_valid_pitch, last_valid_roll, last_valid_throttle, yaw_desired

    pitch = 0
    roll = 0
    throttle = 0
    current_time = robot.getTime()

    # Gestión del estado de centrado en roll
    if estado == "centrado_roll":
        if marker_detected:
            distancia_minima_roll = -200
            distancia_maxima_roll = 150
            roll_max = 0.3
            roll_min = 0.1

            if roll_contador_cero >= 16:
                print("Roll mantenido en 0 después de 16 veces consecutivas. Iniciando control de pitch.")
                roll_objetivo = 0
                roll_contador_cero = 0
                estado = "centrado_throttle"
            else:
                if tvec[0] > distancia_maxima_roll:
                    diferencia = tvec[0] - distancia_maxima_roll
                    roll_objetivo = -max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif tvec[0] < distancia_minima_roll:
                    diferencia = distancia_minima_roll - tvec[0]
                    roll_objetivo = max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    roll_objetivo = 0

                if distancia_minima_roll <= tvec[0] <= distancia_maxima_roll:
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
            set_roll(roll)
            print(f"Dron moviéndose con un roll de {roll:.3f}.")
        else:
            estado = 'mapping'

    elif estado == "centrado_throttle":
        if marker_detected:
            distancia_minima_throttle = -200
            distancia_maxima_throttle = 100
            throttle_max = 0.2
            throttle_min = 0.1

            if contador_throttle_cero >= 16:
                print("throttle mantenido en 0 después de 16 veces consecutivas. Iniciando control de pitch.")
                throttle_objetivo = 0
                contador_throttle_cero = 0
                estado = "control_pitch"
            else:
                if tvec[1] > distancia_minima_throttle:
                    diferencia = tvec[1] - distancia_minima_throttle
                    throttle_objetivo = -max(throttle_min, min(throttle_max, throttle_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif tvec[1] < distancia_maxima_throttle:
                    diferencia = distancia_maxima_throttle - tvec[1]
                    throttle_objetivo = max(throttle_min, min(throttle_max, throttle_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    throttle_objetivo = 0

                if distancia_minima_throttle <= tvec[1] <= distancia_maxima_throttle:
                    contador_throttle_cero += 1
                    print("Dron centrado en el eje Y, throttle ajustado a 0.")
                else:
                    contador_throttle_cero = 0

            last_marker_seen_time = current_time        
            last_valid_throttle = throttle_objetivo  # Guardar el último roll válido
        else:
            throttle_objetivo = last_valid_throttle

        # Suavizar la transición del roll actual al roll objetivo
        suavizado = 0.15  # Factor de suavizado, ajusta según sea necesario
        throttle = throttle_actual + (throttle_objetivo - throttle_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_throttle(throttle)
            print(f"Dron moviéndose con un throttle de {throttle:.3f}.")
        else:
            estado = 'mapping'

    # Gestión del estado de control de pitch
    elif estado == "control_pitch":
        if marker_detected:
            distancia_minima = 2000
            distancia_maxima = 2150
            pitch_max = 0.3  # Reducción del pitch máximo para suavizar el movimiento
            pitch_min = 0.1  # Pitch mínimo

            if contador_pitch_cero >= 16:
                print("Pitch mantenido en 0 después de 16 veces consecutivas.")
                pitch_objetivo = 0
            else:
                if distancia_a_camara > distancia_maxima:
                    diferencia = distancia_a_camara - distancia_maxima
                    pitch_objetivo = max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                elif distancia_a_camara < distancia_minima:
                    diferencia = distancia_minima - distancia_a_camara
                    pitch_objetivo = -max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                else:
                    pitch_objetivo = 0

                if distancia_minima <= distancia_a_camara <= distancia_maxima:
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
            set_pitch(pitch)
            print(f"Dron moviéndose con un pitch de {pitch:.3f}.")
        else:
            estado = 'mapping'

    # Gestión del estado de mapeo
    elif estado == 'mapping':
        set_pitch(0)
        set_roll(0)
        set_throttle(0)
        yaw_desired = 0.6  # Ajuste continuo del yaw
        set_yaw(yaw_desired)
        print("Marcador no detectado por más de 3 segundos. Ajustando yaw para buscar marcador.")
        if marker_detected:
            print('Marcador detectado nuevamente.')
            estado = 'centrado_roll'
            set_yaw(0)

    return pitch, roll, throttle, estado
if __name__ == '__main__':
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)
    door_motor = robot.getDevice('door_motor')

    # Establecer la posición de destino para abrir la puerta (en radianes)
    open_position = 1.57  # 90 grados
    
    # Establecer la posición de destino para cerrar la puerta (en radianes)
    closed_position = 0.0
    
    # Función para abrir la puerta
    def open_door():
        door_motor.setPosition(open_position)
    
    # Función para cerrar la puerta
    def close_door():
        door_motor.setPosition(closed_position)
    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Initialize variables
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    landing_mode = False
    takeoff_mode = False
    contador_pitch_cero = 0
    roll_contador_cero = 0
    contador_throttle_cero = 0
    last_sideways_change_time = 0
    current_sideways = 0
    pitch_actual = 0
    roll_actual = 0
    throttle_actual = 0
    last_marker_seen_time = 0
    last_valid_pitch = 0  # Inicializar el último pitch válido
    last_valid_roll = 0
    last_valid_throttle = 0
    estado = "mapping"  # Estado inicial

    print("\n")
    print("====== Controls =======\n\n")
    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n ")
    print("- Use W and S to go up and down\n ")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n")
    print("- Press T to take off\n")
    print("- Press L to land the drone\n")
    # Main loop:
    while robot.step(timestep) != -1:
        if not landing_mode and not takeoff_mode:
            dt = robot.getTime() - past_time
            actual_state = {}

            if first_time:
                past_x_global = gps.getValues()[0]
                past_y_global = gps.getValues()[1]
                past_time = robot.getTime()
                first_time = False

            # Get sensor data
            roll = imu.getRollPitchYaw()[0]
            pitch = imu.getRollPitchYaw()[1]
            yaw = imu.getRollPitchYaw()[2]
            yaw_rate = gyro.getValues()[2]
            x_global = gps.getValues()[0]
            v_x_global = (x_global - past_x_global) / dt
            y_global = gps.getValues()[1]
            v_y_global = (y_global - past_y_global) / dt
            altitude = gps.getValues()[2]

            # Get body fixed velocities
            cos_yaw = cos(yaw)
            sin_yaw = sin(yaw)
            v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
            v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

            # Initialize values
            desired_state = [0, 0, 0, 0]
            forward_desired = 0
            sideways_desired = 0
            yaw_desired = 0
            height_diff_desired = 0

            key = keyboard.getKey()
            while key > 0:
                if key == Keyboard.UP:
                    forward_desired += 0.5
                elif key == Keyboard.DOWN:
                    forward_desired -= 0.5
                elif key == Keyboard.RIGHT or key == Keyboard.LEFT:
                    new_direction = -1 if key == Keyboard.RIGHT else 1
                    if current_sideways != new_direction:
                        if time.time() - last_sideways_change_time >= MIN_TIME_BETWEEN_DIRECTION_CHANGES:
                            sideways_desired = -0.5 if key == Keyboard.RIGHT else 0.5
                            last_sideways_change_time = time.time()
                            current_sideways = new_direction
                    else:
                        sideways_desired = -0.5 if key == Keyboard.RIGHT else 0.5
                elif key == ord('Q'):
                    yaw_desired = + 1
                elif key == ord('E'):
                    yaw_desired = - 1
                elif key == ord('W'):
                    height_diff_desired = 0.1
                elif key == ord('S'):
                    height_diff_desired = - 0.1
                elif key == ord('A'):
                    if not autonomous_mode:
                        autonomous_mode = True
                        print("Autonomous mode: ON")
                elif key == ord('D'):
                    if autonomous_mode:
                        autonomous_mode = False
                        print("Autonomous mode: OFF")
                elif key == ord('L'):
                    print("Landing initiated.")
                    landing_mode = True
                    break
                elif key == ord('T'):
                    print("Takeoff initiated.")
                    takeoff_mode = True
                    break
                elif key == ord('K'):
                    open_door()
                key = keyboard.getKey()

            height_desired += height_diff_desired * dt

            # Process camera image with OpenCV
            camera_image = camera.getImage()
            height = camera.getHeight()
            width = camera.getWidth()
            image = np.frombuffer(camera_image, np.uint8).reshape((height, width, 4))
            image_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

            # Detección de marcadores ArUco
            corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            marker_detected = len(corners) > 0
            
            if marker_detected:
                rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)
                if ids is not None:
                    for i, id in enumerate(ids):
                        # Dibujar cuadrado y ID
                        cv.aruco.drawDetectedMarkers(image_bgr, corners, ids)
                        
                        # Calcular y mostrar la distancia a la cámara
                        tvec = tvecs[i][0]
                        print(tvec)
                        distancia_a_camara = np.linalg.norm(tvec)
                        print(distancia_a_camara)
                        cv.putText(image_bgr, f"ID: {id[0]}, Dist: {distancia_a_camara:.2f}", 
                                (int(corners[i][0][0][0]), int(corners[i][0][0][1])), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cX, cY = int(np.mean(corners[i][0][:, 0])), int(np.mean(corners[i][0][:, 1]))
                        frame_center_x = gray.shape[1] / 2  # Centro x de la imagen
                        frame_center_y = gray.shape[0] / 2  # Centro y de la imagen
                        pitch_actual, roll_actual, throttle_actual, estado = follow_marker(cX, cY, frame_center_x, frame_center_y, distancia_a_camara, pitch_actual, roll_actual, throttle_actual, marker_detected, tvec, estado)
            else:
                pitch_actual, roll_actual, throttle_actual, estado = follow_marker(0, 0, 0, 0, 0, pitch_actual, roll_actual, throttle_actual, marker_detected, [0, 0, 0], estado)

            # Muestra la imagen en una ventana de OpenCV
            cv.imshow("Crazyflie Camera View", image_bgr)
            cv.waitKey(1)

            # PID velocity controller with fixed height
            motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                            yaw_desired, height_desired,
                                            roll, pitch, yaw_rate,
                                            altitude, v_x, v_y)

            m1_motor.setVelocity(-motor_power[0])
            m2_motor.setVelocity(motor_power[1])
            m3_motor.setVelocity(-motor_power[2])
            m4_motor.setVelocity(motor_power[3])

            past_time = robot.getTime()
            past_x_global = x_global
            past_y_global = y_global
        elif landing_mode:
            land()
            landing_mode = False
        elif takeoff_mode:
            takeoff()
            takeoff_mode = False

