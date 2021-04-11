import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL = 'model/power_model'

class PowerDetector():
    def __init__(self):

        self.model = load_model(MODEL, compile=True)

    def detect(self, image, color_order='bgr'):
         
        image = self.__preprocess(image, color_order)
        output = np.argmax(self.model(image, training=False))

        return output

    def __preprocess(self, image, color_order):
        
        if color_order == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.resize(image, (50, 50), interpolation = cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = np.asarray([image])
        image = np.expand_dims(image, axis=3)

        image = image / 255.0
        image = image.astype('float32')

        return image
