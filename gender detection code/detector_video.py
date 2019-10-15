#https://realpython.com/face-detection-in-python-using-a-webcam/

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv


# model path
model_path = "model.h5"
model_weights_path = "weights.h5"

# load model
model = load_model(model_path)
model.load_weights(model_weights_path)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces, confidences = cv.detect_face(frame)

    for face in faces:
        (startX,startY) = face[0],face[1]
        (endX,endY) = face[2],face[3]
        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,100,0), 2)

        # preprocessing for gender detection model
        cropped_face = frame[startY:endY,startX:endX]
        
        if (cropped_face.shape[0]) < 10 or (cropped_face.shape[1]) < 10:
            continue

        cropped_face = cv2.resize(cropped_face, (150,150))
        cropped_face = cropped_face.astype("float32") / 255
        cropped_face = img_to_array(cropped_face)
        cropped_face = np.expand_dims(cropped_face, axis=0)

        # apply prediction
        conf = model.predict(cropped_face)[0]

        if conf[0] > conf[1]:
            label = "Male"
        else:
            label = "Female"

        print(conf)
        cv2.putText(frame, label, (startX, startY-5),  cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,100,0), 2)

    # Show result
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()