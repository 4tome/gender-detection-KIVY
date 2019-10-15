#https://www.programcreek.com/python/example/89389/cv2.resize
#https://towardsdatascience.com/implement-face-detection-in-less-than-3-minutes-using-python-9f6b43bb3160

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# model path
model_path = "model.h5"
model_weights_path = "weights.h5"

image_path = 'white2.jpg'
im = cv2.imread(image_path)

# load model
model = load_model(model_path)
model.load_weights(model_weights_path)


faces, confidences = cv.detect_face(im)
# loop through detected faces and add bounding box
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    # draw rectangle over face
    cv2.rectangle(im, (startX,startY), (endX,endY), (0,100,0), 2)

    # preprocessing for gender detection model
    cropped_face = im[startY:endY,startX:endX]
    cropped_face = cv2.resize(cropped_face, (150,150))
    cropped_face = cropped_face.astype("float32") / 255
    cropped_face = img_to_array(cropped_face)
    cropped_face = np.expand_dims(cropped_face, axis=0)

    # apply model
    conf = model.predict(cropped_face)[0]

    if conf[0] > conf[1]:
        if conf[0] - conf[1] < conf[1]:
            label = "Maybe male"
        else:
            label = "Male"
    else:
        if conf[0] < conf[1]:
            if conf[1] - conf[0] < conf[0]:
                label = "Maybe female"
            else:
                label = "Female"

    print(conf)

    if label.find("Maybe") >= 0: 
        cv2.putText(im, label, (startX, startY-5),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (200,0,0), 2)
    else:
        cv2.putText(im, label, (startX, startY-5),  cv2.FONT_HERSHEY_SIMPLEX,1, (200,0,0), 2) 

# display result        
cv2.imshow("imagen", im)

# press any key to close window           
cv2.waitKey()
