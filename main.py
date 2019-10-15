import cv2
import os
import cvlib as cv
import threading
import numpy as np

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivymd.theming import ThemeManager
from kivy.properties import ObjectProperty
from kivy.uix.modalview import ModalView
from kivy.core.window import Window

from keras.preprocessing.image import img_to_array
from keras.models import load_model


def toast(text):
    from kivymd.toast.kivytoast import toast

    toast(text)

class MyGrid(Widget):
    img_path = ObjectProperty(None)
    msg = ObjectProperty(None)

    def file_manager_open(self):
        from kivymd.uix.filemanager import MDFileManager

        self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
        self.file_manager = MDFileManager(
                exit_manager=self.exit_manager,
                select_path=self.select_path,
                previous="",
            )
        self.manager.add_widget(self.file_manager)
        self.file_manager.show(os.path.expanduser("~/Desktop"))
        self.manager_open = True
        self.manager.open()


    def select_path(self, path):
        """It will be called when you click on the file name
        or the catalog selection button.
        :type path: str;
        :param path: path to the selected directory or file;
        """

        self.exit_manager()
        toast(path)
        self.img_path.text = path
        # Call some method that may take a while to run.
        # I'm using a thread to simulate this
        #mythread = threading.Thread(target=self.gender_detector(path))
        #mythread.start()

    def exit_manager(self, *args):
        """Called when the user reaches the root of the directory tree."""
        self.manager.dismiss()
        self.manager_open = False

    def process_button_click(self, index):

        if index == 1:
            mythread = threading.Thread(target=self.gender_detection())
            mythread.start()

        if index == 2:
            mythread = threading.Thread(target=self.gender_detection_video())
            mythread.start()

    def gender_detection(self):

        image_path = self.img_path.text

        # Path of the gender detection model
        model_path = "model.h5"
        model_weights_path = "weights.h5"

        im = cv2.imread(image_path)

        if im is None:
            print("Could not read input image")
            popup = Popup(title='No image found',
                          content=Label(text="Could not read input image.\nMake sure you selected an image."),
                          size_hint=(None, None), size=(400, 200))
            popup.open()
            return


        # Load the trained model
        model = load_model(model_path)
        model.load_weights(model_weights_path)


        faces, confidences = cv.detect_face(im)

        # loop through detected faces and add bounding box
        for face in faces:
            (startX,startY) = face[0],face[1]
            (endX,endY) = face[2],face[3]

            # draw rectangle over face
            cv2.rectangle(im, (startX,startY), (endX,endY), (232, 145,15), 2)

            # Preprocessing for the detection
            cropped_face = im[startY:endY,startX:endX]
            cropped_face = cv2.resize(cropped_face, (150,150))
            cropped_face = cropped_face.astype("float32") / 255
            cropped_face = img_to_array(cropped_face)
            cropped_face = np.expand_dims(cropped_face, axis=0)

            # Apply prediction to the cropped face
            conf = model.predict(cropped_face)[0]

            if conf[0] > conf[1]:
                label = "Male"
            else:
                label = "Female"

            print(conf)

            # Print label above the rectangle
            cv2.putText(im, label, (startX, startY-5),  cv2.FONT_HERSHEY_SIMPLEX,1, (232, 145,15), 2)

        # Display output
        cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)

        #imS = cv2.resize(im, (960, 540))  // This is in case you want to resize the output image
        cv2.imshow("output", im)

        # Press any key to close window
        cv2.waitKey()

    def gender_detection_video(self):

        model_path = "model.h5"
        model_weights_path = "weights.h5"

        model = load_model(model_path)
        model.load_weights(model_weights_path)

        video_capture = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces, confidences = cv.detect_face(frame)

            for face in faces:
                (startX, startY) = face[0], face[1]
                (endX, endY) = face[2], face[3]

                # Draw rectangle over face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (232, 145,15), 2)

                # Preprocessing for the detection
                cropped_face = frame[startY:endY, startX:endX]

                if (cropped_face.shape[0]) < 10 or (cropped_face.shape[1]) < 10:
                    continue

                cropped_face = cv2.resize(cropped_face, (150, 150))
                cropped_face = cropped_face.astype("float32") / 255
                cropped_face = img_to_array(cropped_face)
                cropped_face = np.expand_dims(cropped_face, axis=0)

                # Apply prediction to the cropped face
                conf = model.predict(cropped_face)[0]

                if conf[0] > conf[1]:
                    label = "Male"
                else:
                    label = "Female"

                print(conf)

                # Print label above the rectangle
                cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (232, 145,15), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

class GenderDetectionApp(App): # <- Main Class
    title = "Gender detection"
    theme_cls = ThemeManager()
    theme_cls.theme_style = "Dark"
    theme_cls.primary_palette = "BlueGray"

    Window.size = (780, 320)

    def build(self):
        return MyGrid()


if __name__ == "__main__":
    GenderDetectionApp().run()