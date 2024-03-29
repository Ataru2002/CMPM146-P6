from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential(
            [
                layers.Input(shape=input_shape),
                Rescaling(1./255),
                layers.Conv2D(8, kernel_size=(3, 3), input_shape=input_shape, activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), input_shape=input_shape,activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                #layers.Dropout(0),
                layers.Flatten(),
                layers.Dropout(0.5),
                #layers.Dense(16, activation='relu'),
                layers.Dense(categories_count, activation="softmax"),
            ]
        )
        


    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer = RMSprop(learning_rate = 0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'],
        )
        #self.model.print_summary()
