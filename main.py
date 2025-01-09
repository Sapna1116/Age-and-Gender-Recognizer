#import opencv-python
import cv2

#To draw the facebox
def facebox(faceNet,frame):
    #taking the frame
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    #preprocessing the frame
    #After getting the face boxes, we create a 4-dimensional blob from the image. In doing this, we scale it, resize it, and pass in the mean values.
    #A fn to pre-process the image and gives it to cnn for prediction
    blob=cv2.dnn.blobFromImage(frame, 1.0,(227,227),[104,117,123],swapRB=False)
    #Give input to the "net"
    faceNet.setInput(blob)
    #process and give output for the net 
    detection=faceNet.forward()
    #but the output is not visible to us, so we will have to draw a rect. around the img to detect the position of the face 
    bboxs=[]  #face rect.
    for i in range(detection.shape[2]):
        #detection[0, 0, i] give all the 1-D arrays
        #4th-bracket , 3rd bracket, particular i's in 2nd bracket, 1st bracket w/ 7 vals.
        confidence=detection[0,0,i,2]   
        #if confidence val. is greater than confidence threshold val. then only show the result
        if confidence>0.7:
            # 7 things in detection (4-d array)
            #detection = [[[[0, 1, confidenceVal, x1, y1, x2, y2]]]]
            x1=int(detection[0, 0, i, 3] * frameWidth)
            y1=int(detection[0, 0, i, 4] * frameHeight)
            x2=int(detection[0, 0, i, 5] * frameWidth)
            y2=int(detection[0, 0, i, 6] * frameHeight)
            #Creating the rect.
            bboxs.append([x1,y1,x2,y2])
            #drawing the rect.
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1) #(img, corner-1, corener-2, color, round=(int(round(frameheight/150))))
    return frame,bboxs

#path of facemodel and proto
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

#path of agemodel and proto
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

#path of agemodel and proto
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

#to load our models and protos
faceNet=cv2.dnn.readNet(faceModel,faceProto)
#"readNet can read from caffee, darknet, modeloptimizer, tensorflow.torch, etc"
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

#Mean value of model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#img is trained like this only
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

#Now, initialize the video stream and allow the camera sensor to warm up.
video=cv2.VideoCapture(0)

padding=20

#running a infinity loop to keep camera on
while True:
    ret,frame=video.read()
    if not ret :
        cv2.waitKey()
        break
    
    #Fn call
    #resultImg, faceboxes
    frame,bboxs=facebox(faceNet,frame)
    if not bboxs :
        print("No face is detected")
    for bbox in bboxs:
        #Extra pre-processing
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2 ] + padding, frame.shape[1] - 1)]
        blob=cv2.dnn.blobFromImage(frame, 1.0,(227,227),MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPrediction=genderNet.forward()
        gender=genderList[genderPrediction[0].argmax()]

        ageNet.setInput(blob)
        agePrediction = ageNet.forward()
        #Because for each range of probability of its trueness came and to find a particular age range we will find the range with max prob.
        age = ageList[agePrediction[0].argmax()]

        label="{},{}".format(gender,age)
        #To show the rect. and the labels
        #(img, text, origin(1st corner points), fontface, fontscale(width), color[, thickness[,lineType[, bottonLeftOrigin]])
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    #to exit the program user has to press q
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()