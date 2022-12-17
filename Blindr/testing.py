import cv2
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
from Calibration import calibration
def image():
    model = keras.models.load_model('model03.h5',compile=False)
    test_img = cv2.imread(r"D:\Miniproject\app for visual impaired\augmentation\video345.jpg",cv2.IMREAD_COLOR)
    test_img = cv2.resize(test_img,(516,516))
    # test_img = test_img[0:256,0:126]
    # test_img = cv2.resize(test_img,(256,256))
    cv2.imshow("test_img",test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    #test_img = cv2.cvt
    # test_img = 
    prediction = model.predict(test_img)
    print(prediction)
    prediction_image = prediction.reshape((256,256))
    plt.imshow(prediction_image,cmap = 'gray')
    cv2.imshow("prediction",prediction_image)
    cv2.waitKey(0)


class video:
    def lineforalert(self,pred_frame,flag):
        line_val = 128
        # pred_frame = 
        
        if flag == True:
            cv2.line()
    def start(self):
        ############################### ##############################
        model = keras.models.load_model(r'model03.h5',compile=False)
        ############################### ##############################
        result = cv2.VideoWriter('pred_video.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (256,256))
        #############################################################
        cap = cv2.VideoCapture(r"drive-download-20221121T062147Z-001\VID_20221117_144024.mp4")
        # C:\Users\anklesharora\Desktop\Blindr2\Blindr\vid.mp4
        cal = calibration()
        points = [[97,157],[29,256],[159,157],[223,256]]
        croping_para = int((points[0][1]+ points[2][1])/2)
        mid_line = int((points[0][0]+ points[2][0])/2)
        ########################################################
        frame_count = 0
        error_count=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            test_img = cv2.resize(frame,(256,256))
            rot_img =  cv2.rotate(test_img,cv2.ROTATE_90_CLOCKWISE)
            resized_rot_image = cv2.resize(rot_img,(700,700))
            cv2.imshow("frame",rot_img)
            frame = np.expand_dims(rot_img, axis = 0)
########################################################################
            prediction = model.predict(frame)
########################################################################
            prediction_image = prediction.reshape((256,256))
            # rot_img_pred =  cv2.rotate(prediction_image,cv2.ROTATE_90_CLOCKWISE)
            resizes_rot_img_pred = cv2.resize(prediction_image,(700,700))
            rgb =  cv2.cvtColor(prediction_image, cv2.COLOR_GRAY2RGB)
            #a boolean value for alerting 
            # if self.lineforalert():
            
                # vibrate and make sound
            cal.drawing(points,prediction_image)
            print(cal.area(rgb,croping_para,mid_line,points))
            cv2.imshow("prediction",prediction_image)
            # result.write(rot_img_pred)
            
            frame_count+=1
            if cv2.waitKey(0) == ord("e"):
                continue
            elif cv2.waitKey(0) == ord("w"):
                error_count+=1
                continue  
            elif cv2.waitKey(0) == ord("q"):
                break
        print("error count",error_count)
        print("total frame :",frame_count)
        print("Accuracy percentage :",(1-(error_count/frame_count))*100)
        cap.release()
        cv2.destroyAllWindows()

if "__main__" == __name__ : 
    vid = video()
    vid.start()