
import cv2
from flask import Flask, render_template, Response, request, flash
import torch
import mediapipe as mp


mpose = mp.solutions.pose
pose = mpose.Pose()
mpdraw = mp.solutions.drawing_utils


global grey, blur, detect, posed, iron, hulk
grey=0
blur=0
detect =0
posed = 0
iron =0
hulk =0

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom



app = Flask(__name__, template_folder='./templates')

cap = cv2.VideoCapture(0)

def mask(frame, image, l_shoulder_x, l_shoulder_y):
    rol = cv2.imread(image, -1)
    rol = cv2.resize(rol, (300, 300), cv2.INTER_AREA)
    frame_h, frame_w, frame_c = frame.shape
    rol_h, rol_w, rol_c = rol.shape
    rol = cv2.cvtColor(rol, cv2.COLOR_BGR2BGRA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for i in range(rol_h):
        for j in range(rol_w):
            if rol[i, j][3] != 0:
                if (l_shoulder_y - int(rol_h / 2)) + i >= 0:
                    if (l_shoulder_y - int(rol_h / 2)) + i < frame_h:
                        if (l_shoulder_x - int(rol_w / 2)) + j >= 0:
                            if (l_shoulder_x - int(rol_w / 2)) + j < frame_w:
                                frame[(l_shoulder_y - int(rol_h / 2)) + i, (l_shoulder_x - int(rol_w / 2)) + j] = rol[
                                    i, j]
    return frame

def stream():
    l_shoulder_x = 0
    l_shoulder_y = 0
    angle = 0
    while True:
        suc, frame = cap.read()
        if suc:
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (blur):
                frame = cv2.blur(frame, (10,10))

            if (detect):
                result = model(frame)
                for i in range(result.xyxy[0].numpy().shape[0]):

                    if result.xyxy[0].numpy()[i][5] == 0:
                        xmin = int(result.xyxy[0].numpy()[i][0])
                        ymin = int(result.xyxy[0].numpy()[i][1])
                        xmax = int(result.xyxy[0].numpy()[i][2])
                        ymax = int(result.xyxy[0].numpy()[i][3])
                        start_point = (xmin, ymin)
                        end_point = (xmax, ymax)
                        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
            if (posed):
                imcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = pose.process(imcolor)
                h, w = frame.shape[:2]
                lm = result.pose_landmarks
                if lm:
                    lm_pose = mpose.PoseLandmark
                    l_shoulder_x = int(lm.landmark[lm_pose.NOSE].x * w)
                    l_shoulder_y = int(lm.landmark[lm_pose.NOSE].y * h)
                    cv2.circle(frame, (l_shoulder_x, l_shoulder_y), radius=5, color=(0, 0, 0), thickness=-1)

            if (iron and posed ):
                frame = mask(frame, 'iron.png', l_shoulder_x, l_shoulder_y)

            if (hulk and posed):
                frame = mask(frame, 'Hulk.png', l_shoulder_x, l_shoulder_y)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()


            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods =['POST', 'GET'])
def tasks():
    if request.method =='POST':
        if request.form.get('grey') =='Grey':
            global grey
            grey = not grey
        if request.form.get('blur') =='Blur':
            global blur
            blur = not blur
        if request.form.get('detect') =='Detect':
            global detect
            detect = not detect
        if request.form.get('pose') =='Pose':
            global posed
            posed = not posed
        if request.form.get('iron') =='Ironman':
            global iron
            iron = not iron
        if request.form.get('hulk') == 'Hulk':
            global hulk
            hulk = not hulk
    return render_template('index.html')
if __name__ == '__main__':
    app.run()
