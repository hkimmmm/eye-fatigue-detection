import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("sys.path:", sys.path) 

from tensorflow.keras.models import load_model
from public.preprocessing.preprocessing import preprocess_dataset

# Load the trained model from the .h5 file
model = load_model('models/eye_state_model.h5')

# Preprocess the test dataset
images_test, labels_test = preprocess_dataset('data/test')

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(images_test, labels_test, batch_size=32)

# Print the evaluation results
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
