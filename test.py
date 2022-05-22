import numpy as np
import cv2
import pickle

######################
width=640
height=480
threshold=0.65
######################
cap= cv2.VideoCapture(1)
cap.set(3,width)
cap.set(4,height)

pickle_in= open("A/model_trained_10epoch.p", "rb")
model= pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


cap = cv2.VideoCapture()
# The device number might be 0 or 1 depending on the device and the webcam
cap.open(0, cv2.CAP_DSHOW)
while(True):
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow('Processed image', img)
    img = img.reshape(1,32,32,1)
    #Predict
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal = np.amax(predictions)
    print(classIndex,probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal,str(classIndex) + "  " + str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
    cv2.imshow("OriginalImage",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()