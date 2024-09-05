import cv2
import torch
import numpy as np
import matplotlib.path as mplPath
import time
import requests

# Configura el token de tu bot y el chat_id del grupo
BOT_TOKEN = 'token'
CHAT_ID = ''


DETECTION_THRESHOLD = 2  # Segundos necesarios para confirmar detección
RECORDING_DURATION_AFTER_LOSS = 10  # Segundos adicionales de grabación tras perder la detección
MISSED_DETECTION_TOLERANCE = 0.5  # Tolerancia de tiempo para detección perdida

def get_center(bbox):
    # xmin, ymin, xmax, ymax
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center

def load_model():
    print("Loading model...")
    model = torch.hub.load("ultralytics/yolov5", model="yolov5n", pretrained=True)
    print("Model loaded successfully!")
    return model

def get_bboxes(preds: object):
    # xmin, ymin, xmax, ymax, confianza, nombre
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.3]
    return df[["xmin", "ymin", "xmax", "ymax", "confidence", "name"]]


def send_alert(video_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": "¡Alerta! Se detectó una persona."
    }
    requests.post(url, data=data)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    with open(video_path, 'rb') as video:
        data = {
            "chat_id": CHAT_ID
        }
        files = {
            "video": video
        }
        requests.post(url, data=data, files=files)

def detector(frame, model, last_detection_time, start_record_time, recording, video_writer):
    preds = model(frame)
    bboxes = get_bboxes(preds)

    current_time = time.time()
    person_detected = False
    
    # Verifica si hay detección de persona en el frame actual
    for _, row in bboxes.iterrows():
        box = row[["xmin", "ymin", "xmax", "ymax"]].astype(int)
        xc, yc = get_center(box)
        
        if row["name"] == "person":
            person_detected = True
            # Solo actualizamos last_detection_time si es la primera detección
            if not recording and last_detection_time == 0:
                last_detection_time = current_time
        
        # Dibujar las cajas y centros
        cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0,255,0), thickness=-1)
        cv2.rectangle(img=frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(255, 0, 0), thickness=1)
        label = f"{row['name']}: {row['confidence']:.2f}"
        cv2.putText(img=frame, text=label, org=(box[0], box[1] - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0), thickness=1)

    # Lógica para iniciar la grabación
    if not recording and person_detected:
        if last_detection_time == 0:
            last_detection_time = current_time  # Inicializamos el tiempo de detección
        elif (current_time - last_detection_time) >= DETECTION_THRESHOLD:
            # Si se detecta una persona por más de DETECTION_THRESHOLD segundos, empieza a grabar
            recording = True
            start_record_time = current_time
            video_writer = cv2.VideoWriter('detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))

    # Lógica para mantener la grabación
    if recording:
        video_writer.write(frame)
        if person_detected:
            last_detection_time = current_time  # Reinicia el tiempo de detección si sigue detectando personas
        else:
            # Si no se detecta a una persona, verifica si el tiempo transcurrido está dentro de la tolerancia
            if (current_time - last_detection_time) > RECORDING_DURATION_AFTER_LOSS:
                # Termina la grabación si no se detectan personas durante más del tiempo permitido
                recording = False
                video_writer.release()
                last_detection_time = 0
                send_alert('detection.mp4')

    return frame, last_detection_time, start_record_time, recording, video_writer



def main():
    cap = cv2.VideoCapture(0)
    model = load_model()

    last_detection_time = 0
    start_record_time = 0
    recording = False
    video_writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, last_detection_time, start_record_time, recording, video_writer = detector(
            frame, model, last_detection_time, start_record_time, recording, video_writer
        )

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
