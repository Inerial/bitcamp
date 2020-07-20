from flask import Flask, render_template, Response
# emulated camera
import cv2, numpy as np
from threading import Thread
import cv2

YOLO_net = cv2.dnn.readNet("C:\\Users\\bitcamp\\darkflow-master\\bin\\yolov2.weights","C:\\Users\\bitcamp\\darkflow-master\\cfg\\yolo.cfg")

# YOLO NETWORK 재구성
classes = []
with open("C:\\Users\\bitcamp\\darkflow-master\\cfg\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

class WebcamVideoStream:

       def __init__(self, src=0):
           # initialize the video camera stream and read the first frame
           # from the stream
           print("init")
           self.stream = cv2.VideoCapture(src)
           (self.grabbed, self.frame) = self.stream.read()

           # initialize the variable used to indicate if the thread should
           # be stopped
           self.stopped = False


       def start(self):
           print("start thread")
           # start the thread to read frames from the video stream
           t = Thread(target=self.update, args=())
           t.daemon = True
           t.start()
           return self

       def update(self):
           print("read")
           # keep looping infinitely until the thread is stopped
           while True:
               # if the thread indicator variable is set, stop the thread
               if self.stopped:
                   return


               # otherwise, read the next frame from the stream
               (self.grabbed, self.frame) = self.stream.read()

       def read(self):
           # return the frame most recently read
           return self.frame

       def stop(self):
           # indicate that the thread should be stopped
           self.stopped = True

app = Flask(__name__, template_folder='C:\coding\streamingserver\templates')

@app.route('/')
def index():
       """Video streaming home page."""
       return render_template('camera.html')


def gen(camera):
        """Video streaming generator function."""
        while True:
            frame = camera.read()

            h, w, c = frame.shape

            # YOLO 입력
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
            True, crop=False)
            YOLO_net.setInput(blob)
            outs = YOLO_net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            print(1)
            for out in outs:

                for detection in out:

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        dw = int(detection[2] * w)
                        dh = int(detection[3] * h)
                        # Rectangle coordinate
                        x = int(center_x - dw / 2)
                        y = int(center_y - dh / 2)
                        boxes.append([x, y, dw, dh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            print(2)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

            print(3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    score = confidences[i]

                    # 경계상자와 클래스 정보 이미지에 입력
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
                    (255, 255, 255), 1)

            ret, jpeg = cv2.imencode('.jpg', frame)
            
            # print("after get_frame")
            if jpeg is not None:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                print("frame is none")



@app.route('/video_feed')
def video_feed():
       """Video streaming route. Put this in the src attribute of an img tag."""
       return Response(gen(WebcamVideoStream().start()),
                       mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
       app.run(host='127.0.0.1', port=5010, debug=True, threaded=True)