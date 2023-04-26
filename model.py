from sklearn.model_selection import train_test_split
from tensorflow import keras

class MultimodalModel:
    def __init__(self, audio_shape, geospatial_shape, num_classes):
        self.audio_shape = audio_shape
        self.geospatial_shape = geospatial_shape
        self.num_classes = num_classes
        self.model = None
        
    def build(self):
        # Define the DenseNet model architecture for audio features
        base_audio_model = keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=self.audio_shape)
        x_audio = keras.layers.GlobalAveragePooling2D()(base_audio_model.output)
        x_audio = keras.layers.Dense(256, activation='relu')(x_audio)
        x_audio = keras.layers.Dropout(0.5)(x_audio)

        # Define the DenseNet model architecture for geospatial features
        base_geospatial_model = keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=self.geospatial_shape)
        x_geospatial = keras.layers.GlobalAveragePooling2D()(base_geospatial_model.output)
        x_geospatial = keras.layers.Dense(256, activation='relu')(x_geospatial)
        x_geospatial = keras.layers.Dropout(0.5)(x_geospatial)

        # Concatenate the outputs of the two models
        merged = keras.layers.concatenate([x_audio, x_geospatial])

        # Add a fully connected layer and an output layer
        merged = keras.layers.Dense(256, activation='relu')(merged)
        merged = keras.layers.Dropout(0.5)(merged)
        output = keras.layers.Dense(self.num_classes, activation='softmax')(merged)

        # Create the multi-modal model
        self.model = keras.models.Model(inputs=[base_audio_model.input, base_geospatial_model.input], outputs=output)

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train_audio, X_train_geospatial, y_train, batch_size, epochs, validation_split=0.1):
      X_train_audio, X_val_audio, X_train_geospatial, X_val_geospatial, y_train, y_val = train_test_split(
        X_train_audio, X_train_geospatial, y_train, test_size=validation_split, random_state=42)

    # Train the model
      history = self.model.fit([X_train_audio, X_train_geospatial], y_train, batch_size=batch_size, epochs=epochs,
                             validation_data=([X_val_audio, X_val_geospatial], y_val))

    # Print training and validation accuracy
      for i, acc in enumerate(history.history['accuracy']):
         print(f"Epoch {i+1}: Training Accuracy = {acc:.4f}, Validation Accuracy = {history.history['val_accuracy'][i]:.4f}")

      return history

    def evaluate(self, X_test_audio, X_test_geospatial, y_test):
        # Evaluate the model on the test set
        return self.model.evaluate([X_test_audio, X_test_geospatial], y_test)
