import cv2 as cv
import os
from time import sleep

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


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret

url = 'http://192.168.118.57:4747/video'
cap = cv.VideoCapture(url)


while True:
    _, frame = cap.read()
    copyframe = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
    # print(ret)
    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    cv.imshow("frame", frame)
    key = cv.waitKey(1)

    if key == ord("q"):
        break
    if board_detected == True:
        # storing the checker board image
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyframe)
        sleep(0.5)

        print(f"saved image number {n}")
        n += 1  # incrementing the image counter
cap.release()
cv.destroyAllWindows()

print("Total saved Images:", n)
