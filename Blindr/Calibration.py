import cv2
import numpy as np
# have to find points
# have to show points and trapezium
# conditions
# deploy

# res = [[157.52581666910066 ,160.1128524708699 ],[51.13990613326484,7.4809533770353625],[259.0130791460439,160.60939657670812],[357.2124850582412,9.707733040088915 ]]
#draw in opencv

class calibration:
    #  is to calibrate and show the trapezium
    def drawing(self,points,frame):
        # for drawing the trapezium
        color = (0, 255, 0) 
        thickness = 2
        frame = cv2.line(frame, (points[0][0], points[0][1]),(points[1][0], points[1][1]), color, thickness)
        frame = cv2.line(frame, (points[1][0], points[1][1]),(points[3][0], points[3][1]), color, thickness)
        frame = cv2.line(frame, (points[2][0], points[2][1]),(points[3][0], points[3][1]), color, thickness)
        frame = cv2.line(frame, (points[2][0], points[2][1]),(points[0][0], points[0][1]), color, thickness)
        # this resize is only for user.
        # frame = cv2.resize(frame, (400, 500))
        # cv2.imshow("frame",frame)
        # cv2.waitKey(0)
    # continously pass the frame and gray pixel is inside the trapezium or not
    # if any 5-10 pixel 
    def masking_white(self,frame):
        # frame = np.squeeze(frame)
        # print(frame)
        # the function will give the points which are gray
        # print(frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 1])
        upper = np.array([0, 0, 225])
        # Defining mask for detecting color
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)    
        # cv2.imshow("mask", mask)
        # print("mask",len(mask[0]))
        # print("mask",mask)
        noney,nonex = np.nonzero(mask)
        # print("non zero",len(noney),len(nonex))
        center_coordinates = (120, 50)

        #debugging
        # print("start")
        # for i in range(len(nonex)):
        #         frame = cv2.circle(frame, (nonex[i],noney[i]), 1, (255, 0, 0), 1)
        # print("reached")

        # Display Image and Mask
        # cv2.imshow("Image", frame)
        # cv2.imshow("Mask", mask)
        
        # Make python sleep for unlimited time
        # cv2.waitKey(0)
        return nonex,noney
    # def slant_line_separation(self,points,coors):
    #     pass
    def cali(self,frame,cp,mid_line,points,coor_x,coor_y):
        # 
        # croping the image
        # Will provide boolean value whether point is inside or outside
        # frame = frame[cp:]
        # # debugging
        # # self.masking_white(frame)
        # left_frame = frame[:,:mid_line]
        # right_frame = frame[:,mid_line:]
        #
        # self.masking_white(right_frame)
        
        # Debugging
        # sample points
        # conditions
        # print("coor",coor_x,coor_y)
        if coor_y >= cp:
            # print("under cp")
            #calculate m for both the line
            if coor_x <= mid_line:
                # print("left side")
                # left side of trapezium
                # m = (points[1][1] - points[0][1])/(points[1][0]-points[0][0])
                # print(m)
                # print((coor_y + 1.456 * coor_x))
                if (coor_y + 1.456 * coor_x) >= 298.232:
                    return True,"l"
                else:
                    return False,""
                    
            else:
                # right side of trapezium
                # m = (points[1][1] - points[0][1])/(points[1][0]-points[0][0])
                # print(m)
                # print("res",(coor_y - 1.478 * coor_x))
                if (coor_y - 1.478 * coor_x) <= -78.002:
                    return True,"r"
                else:
                    return False,""
        else:            
            return False,""

    def area(self,frame,croping_para,mid_line,points):
        arrx,arry = self.masking_white(frame)
        count = 0
        right= 0
        left = 0
        for i in range(len(arrx)):
            inside , side =  self.cali(frame,croping_para,mid_line,points,arrx[i],arry[i])
            if inside:
                count+=1
                if "r" == side:
                    right+=1
                if "l" == side:
                    left+=1
        print("count",count)
        print("right", right)
        print("left", left)
        # count = 6000
        # left = 3000
        # right = 3000
        percent_20 = 4500
        percent_80 = 8801
        # area should be 20 percent 12573 and less then 50 percentage
        if count >= percent_20 and count < percent_80:
            if left >= percent_20 and left < percent_80:
                return "r"
            if right >= percent_20 and right < percent_80:
                # print("inse")
                return "l"
            if left+right > percent_20:
                return "stop"
            elif left < percent_20 and right < percent_20:
                return False
        elif count > percent_80:
            return "stop"
        else:
            return False
if "__main__" == __name__ :
    from tensorflow import keras
# upperleft lowerleft upperright lowerright
    points = [[97,157],[29,256],[159,157],[223,256]]
    croping_para = int((points[0][1]+ points[2][1])/2)
    mid_line = int((points[0][0]+ points[2][0])/2)
    # img_path = r"drive-download-20221121T062147Z-001\VID_20221117_143749.mp4"
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (256, 256))   
    cal = calibration()
    # cal.drawing(points)

                
    # print(cal.cali(img,croping_para,mid_line,points,30,256))
    # print(cal.cali(img,croping_para,mid_line,points,28,256))
    # print(cal.cali(img,croping_para,mid_line,points,97,157))
    # print(cal.cali(img,croping_para,mid_line,points,255,157))
    # for i in range(len(nonex)):
    # import time
    # start = time.time()
    # print(cal.area(img,croping_para,mid_line,points))
    # print(cal.area(img,croping_para,mid_line,points))
    # print(time.time() - start)
    # check whether any point is in trapezium or not
    # cal.cali()
    # img = cv2.resize(img, (400, 500))   
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    # x: 1135.622  and y: 2457.12155 upperleft
    # x: 370.963   and y: 3914.96005 lowerleft
    # x: 1865.936  and y: 2457.12155  upperright
    # x: 2574.675  and y: 3914.96005  loweright
    vid = cv2.VideoCapture(r"drive-download-20221121T062147Z-001\VID_20221117_143749.mp4")
    model = keras.models.load_model(r'C:\Users\anklesharora\Desktop\Blindr2\Blindr\model03.h5',compile=False)
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        rot_img =  cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(rot_img, (256, 256)) 
        cal.drawing(points, frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()