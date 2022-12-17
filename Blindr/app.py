from itertools import count
from pyexpat import model
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
from io import StringIO 
from matplotlib import pyplot as plt
import time
from PIL import Image
import base64,cv2
import pyshine as ps
from flask_cors import CORS,cross_origin
import imutils
import dlib
from tensorflow import keras
from engineio.payload import Payload
from Calibration import calibration
app = Flask(__name__)
app.config['DEBUG'] = True
cors = CORS(app,resources={r"/api/*":{"origins":"*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
global model, points, croping_para, mid_line, cal
model = keras.models.load_model(r'C:\Users\anklesharora\Desktop\Blindr2\Blindr\model03.h5',compile=False)
points = [[97,157],[29,256],[159,157],[223,256]]
croping_para = int((points[0][1]+ points[2][1])/2)
mid_line = int((points[0][0]+ points[2][0])/2)
cal = calibration()

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
# global pre
# pre = False
def inference(frame,cal,pre):
    test_img = cv2.resize(frame,(256,256))
    rot_img =  cv2.rotate(test_img,cv2.ROTATE_180)
    # cv2.imshow("frame",rot_img)
    frame = np.expand_dims(test_img, axis = 0)

    prediction = model.predict(frame)

    prediction_image = prediction.reshape((256,256))
    rot_img_pred =  cv2.rotate(prediction_image,cv2.ROTATE_180)
    # cv2.imshow("frame",rot_img_pred)
    # cv2.waitKey(1)
    rgb =  cv2.cvtColor(rot_img_pred, cv2.COLOR_GRAY2RGB)
    cal.drawing(points,rot_img_pred)
    print(cal.area(rgb,croping_para,mid_line,points))
    #calibration
    # cv2.imshow("frame",frame)
    # cv2.waitKey(1)
    output = cal.area(rgb,croping_para,mid_line,points)
    if pre != output:
        pre = output
    # output 
    #a boolean value for alerting 
    # if self.lineforalert():

        # vibrate and make sound    cv2.imshow("prediction",rot_img_pred)
    
    return rot_img_pred,pre

@socketio.on('image')
def image(data_image):



    # Process the image frame


    # name = str("frame.jpg")
    frame = (readb64(data_image))
    
    frame = imutils.resize(frame, width=147)
    frame1 = cv2.flip(frame, 1)
    pre = False
    dim = (700,700 )
    raw = cv2.resize(frame1, dim)
    cv2.imshow("raw",raw)
    frame,sendOp = inference(frame1, cal,pre)
    frame = cv2.resize(frame, dim)
    cv2.imshow("frame",frame)
    cv2.waitKey(0)
    # cv2.imwrite(name,frame)
    # imgencode = cv2.imencode('.jpg', frame)[1]
    

    # base64 encode
    # stringData = base64.b64encode(imgencode).decode('utf-8')
    # b64_src = 'data:image/jpg;base64,'
    # stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', {'data': sendOp})





if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port = '5000')
    # 127.0.0.1



If someone is aware of a visually impaired organization, please dm me.