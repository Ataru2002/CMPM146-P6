from config import BOARD_SIZE, categories, image_size
from tensorflow.keras import models
from models.basic_model import BasicModel
from models.model import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()

class UserInputPlayer:
    def get_move(self, board_state):
        inp = input('Enter x y:')
        try:
            x, y = inp.split()
            x, y = int(x), int(y)
            return x, y
        except Exception:
            return None

import random

class RandomPlayer:
    def get_move(self, board_state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state[i][j] is None:
                    positions.append((i, j))
        return random.choice(positions)

from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2

class UserWebcamPlayer:
    def _process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width, height = frame.shape
        size = min(width, height)
        pad = int((width-size)/2), int((height-size)/2)
        frame = frame[pad[0]:pad[0]+size, pad[1]:pad[1]+size]
        return frame

    def _access_webcam(self):
        import cv2
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
            frame = self._process_frame(frame)
        else:
            rval = False
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            frame = self._process_frame(frame)
            key = cv2.waitKey(20)
            if key == 13: # exit on Enter
                break

        vc.release()
        cv2.destroyWindow("preview")
        return frame

    def _print_reference(self, row_or_col):
        print('reference:')
        for i, emotion in enumerate(categories):
            print('{} {} is {}.'.format(row_or_col, i, emotion))
    
    def _get_row_or_col_by_text(self):
        try:
            val = int(input())
            return val
        except Exception as e:
            print('Invalid position')
            return None
    
    def _get_row_or_col(self, is_row):
        try:
            row_or_col = 'row' if is_row else 'col'
            self._print_reference(row_or_col)
            img = self._access_webcam()
            emotion = self._get_emotion(img)
            if type(emotion) is not int or emotion not in range(len(categories)):
                print('Invalid emotion number {}'.format(emotion))
                return None
            print('Emotion detected as {} ({} {}). Enter \'text\' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.'.format(categories[emotion], row_or_col, emotion))
            inp = input()
            if inp == 'text':
                return self._get_row_or_col_by_text()
            return emotion
        except Exception as e:
            # error accessing the webcam, or processing the image
            raise e
    
    def _get_emotion(self, img) -> int:
        # Your code goes here
        #
        # img an np array of size NxN (square), each pixel is a value between 0 to 255
        # you have to resize this to image_size before sending to your model
        # to show the image here, you can use:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()
        model = Model.load_model("./results/basic_model_10_epochs_timestamp_1708321651.keras")
        model.print_summary()
        resized_image = np.resize(img, (image_size[0], image_size[1]))
        #resized_image = resized_image/255.0
        
        reshaped = np.reshape(resized_image, (image_size[0], image_size[1],1))
        model.model.predict(reshaped)
 
        # You have to use your saved model, use resized img as input, and get one classification value out of it
        # The classification value should be 0, 1, or 2 for neutral, happy or surprise respectively

        # return an integer (0, 1 or 2), otherwise the code will throw an error
        return 1
        pass
    
    def get_move(self, board_state):
        row, col = None, None
        while row is None:
            row = self._get_row_or_col(True)
        while col is None:
            col = self._get_row_or_col(False)
        return row, col