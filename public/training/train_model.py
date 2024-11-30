import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("sys.path:", sys.path) 

from tensorflow.keras.models import Sequential, load_model
from public.preprocessing.preprocessing import preprocess_dataset
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification (open/closed)
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load preprocessed data (images and labels)
images, labels = preprocess_dataset('data/train')

# Create and train the model
model = create_model()
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
os.makedirs('models', exist_ok=True)
model.save('models/eye_state_model.keras')
