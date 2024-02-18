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
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            #layers.Dense(3, activation='relu'),
            layers.Dense(categories_count, activation="softmax"),
        ]
    )
        # input = layers.Input(shape=input_shape)
        # x = layers.Conv2D(32, (3, 3), activation='relu')(input)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # x = layers.Flatten()(x)
        # x = layers.Dense(128, activation='relu')(x)
        # output_layer = layers.Dense(categories_count, activation='softmax')(x)


    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer = RMSprop(learning_rate = 0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'],
        )
        #self.model.print_summary()
