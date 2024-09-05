import sys
import traceback
import tellopy
import av
import cv2  # for avoidance of pylint error
import numpy
import time

import cv2 as cv
import os
from time import sleep



def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret

def main():

    CHESS_BOARD_DIM = (13, 9)

    n = 0  # image_counter

    # checking if  images dir is exist not, if not then create images directory
    image_dir_path = "images"

    CHECK_DIR = os.path.isdir(image_dir_path)
    # if directory does not exist create
    if not CHECK_DIR:
        os.makedirs(image_dir_path)
        print(f'"{image_dir_path}" Directory is created')
    else:
        print(f'"{image_dir_path}" Directory already Exists.')

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    drone = tellopy.Tello()
    path = '/home/user/tello-ai/tello-yollov7-ia/images/'
    cont = 0

    try:
        drone.connect()
        drone.wait_for_connection(20.0)

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
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                gray = cv.cvtColor(numpy.array(frame.to_image()), cv.COLOR_BGR2GRAY)
                images = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                height, width, channels = images.shape
                print(f"Resolución de la cámara: {width}x{height}")
                cpyimages = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                image, board_detected = detect_checker_board(images, gray, criteria, CHESS_BOARD_DIM)
                # Para capturar fotos del dron
                # if cont%10 == 0:
                #     cv2.imwrite(path + 'IMG_%04d.jpg' % cont, image)    
                # cont += 1
                cv.putText(
                    images,
                    f"saved_img : {n}",
                    (30, 40),
                    cv.FONT_HERSHEY_PLAIN,
                    1.4,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
                cv2.imshow('Original', images)
                cv2.waitKey(1)

                key = cv.waitKey(1)

                if key == ord("q"):
                    break
                if board_detected == True:
                    # storing the checker board image
                    cv.imwrite(f"{image_dir_path}/image{n}.png", cpyimages)
                    sleep(0.5)

                    print(f"saved image number {n}")
                    n += 1  # incrementing the image counter


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

if __name__ == '__main__':
    main()
