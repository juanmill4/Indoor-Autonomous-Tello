import sys
import traceback
import tellopy
import av
import cv2
import numpy as np
import cv2.aruco as aruco
import math
import time
from dectector import detector, load_model


SIZE = 94.5  # Ajusta este tamaño al de tus marcadores reales

# Variables globales para guardar los datos del giroscopio y la dirección actual
gyro_data = None
current_direction = 0.0  # Dirección inicial apuntando al norte (0 grados)
last_time = None  # Variable para guardar el tiempo del último frame
marker_data = {}
stored_distances = {'dist_1': [], 'dist_2': [], 'angle_1': [], 'angle_2': []}

def handler(event, sender, data, **args):
    global gyro_data
    drone = sender
    # if event is drone.EVENT_FLIGHT_DATA:
    #     print(data)
    if event is drone.EVENT_LOG_DATA:
        #print(data)
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

def project_points(points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return projected_points

def draw_panel(img, markers_orientations=None, camera_matrix=None, dist_coeffs=None, yaw=0, mapping_mode=False, stored_distances=None, avg_dist_1=None, avg_dist_2=None):
    img.fill(0)
    angle_radians = np.deg2rad(yaw)
    map_center = (500, 100)
    minimap_radius = 70
    outer_radius = minimap_radius + 10

    # Draw minimap circle
    cv2.circle(img, map_center, minimap_radius, (255, 0, 0), 2)
    cv2.circle(img, map_center, outer_radius, (255, 255, 255), 1)
    # Draw a red dot at the center of the minimap
    cv2.circle(img, map_center, 5, (0, 0, 255), -1)

    if markers_orientations:
        for aruco_id, orientations in markers_orientations.items():
            rvec = orientations['rvec']
            tvec = orientations['tvec']
            roll_deg = orientations['roll']
            distance = np.linalg.norm(tvec)
            angle = np.arctan2(tvec[0][0], tvec[0][2])  # Ángulo en radianes utilizando X e Y
            
            marker_data[aruco_id] = {'distance': distance, 'angle': angle, 'tvec': tvec}
        
            max_distance = 3000
            scaled_distance = int((distance / max_distance) * minimap_radius)
            if distance < max_distance:
        
                # Ajustar la posición del marcador según el yaw
                north_corrected_angle =  angle - yaw - angle_radians
                marker_x = int(map_center[0] + scaled_distance * np.cos(north_corrected_angle))
                marker_y = int(map_center[1] - scaled_distance * np.sin(north_corrected_angle))
        
                # Proyectar puntos para las líneas perpendiculares
                perp_length = 400
                panel_points = np.float32([
                    [0, 0, 0], [0, SIZE, 0], [SIZE, SIZE, 0], [SIZE, 0, 0],
                    [0, 0, -SIZE/10], [0, SIZE, -SIZE/10], [SIZE, SIZE, -SIZE/10], [SIZE, 0, -SIZE/10]
                ])
                img_points = project_points(panel_points, rvec, tvec, camera_matrix, dist_coeffs)
                img_points = np.int32(img_points).reshape(-1, 2)
        
                x0, y0 = img_points[0]
                x1, y1 = img_points[3]
        
                # Calcular la pendiente solo si x1 != x0, de lo contrario establecer a infinito
                if x1 != x0:
                    slope = (y1 - y0) / (x1 - x0)
                else:
                    slope = float('inf')
        
                # Calcular los puntos finales de la línea solo una vez
                length = 100  # Aumentar la longitud para mayor sensibilidad
                if slope != float('inf'):
                    dx = length / np.sqrt(1 + slope ** 2)
                    dy = slope * dx
                else:
                    dx = 0
                    dy = length
        
                height, width, _ = img.shape
                # Calcular los puntos extendidos para las líneas
                line1_p1, line1_p2 = calculate_infinite_line_points(img_points[0], img_points[3], width)
                line2_p1, line2_p2 = calculate_infinite_line_points(img_points[1], img_points[2], width)
        
                # Definir el polígono que conecta los puntos extendidos
                pts = np.array([line1_p1, line1_p2, line2_p2, line2_p1], np.int32)
                pts = pts.reshape((-1, 1, 2))
        
                # Rellenar el polígono
                # cv.fillPoly(img, [pts], (255, 0, 0))
        
                perp_x1, perp_y1 = img_points[0]
                perp_x2, perp_y2 = img_points[3]
        
                # Invertir coordenadas para corregir el modo espejo
                line_x1 = int(marker_x + dx)
                line_y1 = int(marker_y - dy)  # Cambio aquí: -dy en lugar de +dy
                line_x2 = int(marker_x - dx)
                line_y2 = int(marker_y + dy)  # Cambio aquí: +dy en lugar de -dy
        
                # Clip lines to the circle's radius
                line_x1, line_y1 = clip_line_to_circle(map_center, minimap_radius, marker_x, marker_y, line_x1, line_y1)
                line_x2, line_y2 = clip_line_to_circle(map_center, minimap_radius, marker_x, marker_y, line_x2, line_y2)
        
                cv2.line(img, (marker_x, marker_y), (line_x1, line_y1), (255, 0, 255), 2)
                cv2.line(img, (marker_x, marker_y), (line_x2, line_y2), (255, 0, 255), 2)
        
                distance_text = f'{distance:.2f} mm'
                #cv.putText(img, distance_text, (marker_x + 10, marker_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(marker_data)

    
    north_x = int(map_center[0] + outer_radius * np.sin(angle_radians))
    north_y = int(map_center[1] - outer_radius * np.cos(angle_radians))
    cv2.putText(img, 'N', (north_x, north_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if mapping_mode:
        text = "Mapping"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 1, (0, 0, 255), 2)

        if '0' in marker_data and '1' in marker_data:
            stored_distances['dist_1'].append(marker_data['0']['distance'])
            stored_distances['angle_1'].append(marker_data['0']['angle'])
            stored_distances['dist_2'].append(marker_data['1']['distance'])
            stored_distances['angle_2'].append(marker_data['1']['angle'])

    # Calcular las distancias medias si hay datos en stored_distances y no estamos en mapping_mode
    if not mapping_mode and stored_distances['dist_1'] and stored_distances['dist_2']:
        avg_dist_1 = np.mean(stored_distances['dist_1'])
        avg_angle_1 = np.mean(stored_distances['angle_1'])
        avg_dist_2 = np.mean(stored_distances['dist_2'])
        avg_angle_2 = np.mean(stored_distances['angle_2'])
        
        avg_dist_1 = avg_dist_1 / 50
        avg_dist_2 = avg_dist_2 / 50
        
        stored_distances['dist_1'].clear()
        stored_distances['dist_2'].clear()
        stored_distances['angle_1'].clear()
        stored_distances['angle_2'].clear()

    # Condición para dibujar el rectángulo
    if not mapping_mode and avg_dist_1 is not None and avg_dist_2 is not None:
        x_c, y_c = 250, 350

        vertices = [
            (x_c - avg_dist_1, y_c + avg_dist_2),
            (x_c + avg_dist_1, y_c + avg_dist_2),
            (x_c + avg_dist_1, y_c - avg_dist_2),
            (x_c - avg_dist_1, y_c - avg_dist_2),
            (x_c - avg_dist_1, y_c + avg_dist_2)
        ]

        vertices = [(int(x), int(y)) for x, y in vertices]

        for i in range(len(vertices) - 1):
            cv2.line(img, vertices[i], vertices[i + 1], (255, 0, 255), 2)
            mid_x = (vertices[i][0] + vertices[i + 1][0]) // 2
            mid_y = (vertices[i][1] + vertices[i + 1][1]) // 2
            avg_dist_11 = avg_dist_1 * 50
            avg_dist_22 = avg_dist_2 * 50
            if i % 2 == 0:
                label = f'D1: {avg_dist_22:.2f}'
            else:
                label = f'D2: {avg_dist_11:.2f}'
            cv2.putText(img, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        drone_position = (int(x_c), int(y_c))
        cv2.circle(img, drone_position, 5, (0, 0, 255), -1)
        
        # Actualizar la posición del dron en función de los datos de los marcadores detectados
        if marker_data and '0' in marker_data and '1' in marker_data:
            tvec_0 = marker_data['0']['tvec']
            tvec_1 = marker_data['1']['tvec']
            # Aquí se podría usar un promedio ponderado, o cualquier otro método de tu preferencia
            drone_x = (tvec_0[0][0] + tvec_1[0][0]) / 4
            drone_y = (tvec_0[0][2] + tvec_1[0][2]) / 4  # Usando la coordenada Z para y del plano

            # Convertir coordenadas de la posición del dron para dibujarlo en el panel (inverso)
            drone_position_x = int(x_c + drone_x / 50)  # Invertir la posición según tus necesidades
            drone_position_y = int(y_c - drone_y / 50)  # Invertir la posición según tus necesidades

            drone_position = (drone_position_x, drone_position_y)
        else:
            drone_position = (int(x_c), int(y_c))

        cv2.circle(img, drone_position, 5, (0, 0, 255), -1)

    return avg_dist_1, avg_dist_2

def calculate_infinite_line_points(p1, p2, img_width):
    # Check for a vertical line
    if p1[0] == p2[0]:
        x1 = x2 = p1[0]
        y1 = 0
        y2 = img_width - 1
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]
        
        x1 = 0
        y1 = int(slope * x1 + intercept)
        
        x2 = img_width - 1
        y2 = int(slope * x2 + intercept)
    
    return (x1, y1), (x2, y2)

def clip_line_to_circle(center, radius, x1, y1, x2, y2):
    # Translate the line to the origin
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - center[0], y1 - center[1]

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        if 0 <= t1 <= 1:
            x2 = int(x1 + t1 * dx)
            y2 = int(y1 + t1 * dy)
        elif 0 <= t2 <= 1:
            x2 = int(x1 + t2 * dx)
            y2 = int(y1 + t2 * dy)
    
    return x2, y2

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Carga la matriz de la cámara y los coeficientes de distorsión
with np.load('/home/user/tello-ai/calib_data_webcam/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

drone = tellopy.Tello()
autonomous_mode = False
mapping_mode = False
initial_yaw = None
yaw_accumulated = 0
avg_dist_1 = None
avg_dist_2 = None
panel_frame = np.zeros((480, 640, 3), dtype=np.uint8)

model = load_model()


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
            
            Detected_ArUco_markers = detect_ArUco(img)
            ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markers, mtx, dist)
            print(ArUco_marker_orientations)
            
            if Detected_ArUco_markers:
                for aruco_id, orientation in ArUco_marker_orientations.items():
                    corners = Detected_ArUco_markers[aruco_id]
                    center = np.mean(corners[0], axis=0)
                    cv2.putText(img, f"ID: {aruco_id}, Yaw: {orientation['yaw']:.2f}, Pitch: {orientation['pitch']:.2f}, Roll: {orientation['roll']:.2f}", 
                                (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    
                    avg_dist_1, avg_dist_2 = draw_panel(panel_frame, ArUco_marker_orientations, mtx, dist, current_direction, mapping_mode, stored_distances, avg_dist_1, avg_dist_2)
            else:
                avg_dist_1, avg_dist_2 = draw_panel(panel_frame, yaw=current_direction, mapping_mode=mapping_mode, stored_distances=stored_distances, avg_dist_1 = avg_dist_1, avg_dist_2 = avg_dist_2)

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

            # Iniciar el modo de mapeo cuando se presione 'M'
            if cv2.waitKey(1) & 0xFF == ord('m'):
                if not mapping_mode:
                    mapping_mode = True
                    initial_yaw = current_direction
                    yaw_accumulated = 0
                    print("Mapping mode: ON")
            
            if cv2.waitKey(1) & 0xFF == ord('t'):
                print('asaksdaskdlaskdlaks')
                drone.takeoff()

            if cv2.waitKey(1) & 0xFF == ord('l'):
                drone.land()

            # Manejo del modo de mapeo
            if mapping_mode:
                drone.set_yaw(0.6)
                yaw_accumulated += abs(gyro_data * delta_time * 180 / np.pi)  # Convertir radianes a grados y acumular
                print(yaw_accumulated)
                if yaw_accumulated >= 2 * 360:  # Giro completo de 360 grados
                    drone.set_yaw(0)
                    mapping_mode = False
                    print("Mapping mode: OFF")
                    yaw_accumulated = 0
                    # Aquí puedes realizar la acción de dibujar el rectángulo basado en las distancias almacenadas

            
            #img = detector(img, model)


            cv2.imshow('Original', img)
            cv2.imshow('Panel 3D', panel_frame)

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
