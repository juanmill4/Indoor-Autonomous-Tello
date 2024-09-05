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
import time

def handler(event, sender, data, **args):
    global prev_flight_data
    global x
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)
        time.sleep(3)
n = 0
SIZE = 94.5

with np.load('/home/user/tello-ai/calib_data/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

drone = tellopy.Tello()

try:
    drone.connect()
    drone.wait_for_connection(20.0)
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
    distance_checks_passed = 0

    while True:
        for frame in container.decode(video=0):
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.time()
            gray = cv.cvtColor(numpy.array(frame.to_image()), cv.COLOR_BGR2GRAY)
            images = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            # Detectar marcadores ArUco
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)
                marker_positions = {}
                marker_centers = {}
                for i, id in enumerate(ids):
                    # Dibujar cuadrado y ID
                    aruco.drawDetectedMarkers(images, corners, ids)
                    
                    # Calcular el centro de la etiqueta ArUco
                    center = np.mean(corners[i][0], axis=0)
                    marker_centers[id[0]] = center
                    
                    # Calcular y mostrar la distancia a la cámara desde el centro de la etiqueta
                    tvec = tvecs[i][0]
                    distancia_a_camara = np.linalg.norm(tvec)
                    print(distancia_a_camara)
                    
                    # Calcular y mostrar el ángulo con respecto a la cámara desde el centro de la etiqueta
                    rvec = rvecs[i][0]
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    yaw, pitch, roll = cv2.decomposeProjectionMatrix(cv2.hconcat((rotation_matrix, np.zeros((3, 1)))))[6]
                    angle_deg = np.degrees(yaw) % 360  # Ajustar para que esté en el rango de 0 a 360 grados
                    print(f"Angle: {angle_deg[0]:.2f}")

                    cv2.putText(images, f"ID: {id[0]}, Dist: {distancia_a_camara:.2f}, Angle: {angle_deg[0]:.2f}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                
                # Dibujar líneas entre los centros de las etiquetas
                if len(marker_centers) > 1:
                    for i in range(len(ids) - 1):
                        for j in range(i + 1, len(ids)):
                            id_i, id_j = ids[i][0], ids[j][0]
                            if id_i in marker_centers and id_j in marker_centers:
                                center_i, center_j = marker_centers[id_i], marker_centers[id_j]
                                distancia = np.linalg.norm(center_i - center_j)
                                midpoint = (center_i + center_j) / 2
                                cv2.line(images, tuple(center_i.astype(int)), tuple(center_j.astype(int)), (255, 0, 0), 2)
                                cv2.putText(images, f"{distancia:.2f}m", tuple(midpoint.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('Original', images)
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
