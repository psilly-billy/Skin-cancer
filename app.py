
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# Load your trained model
model = tf.keras.models.load_model('fine_best_model.h5')

def preprocess_image(image):
    # Preprocess the image to fit model's input requirements
    image = image.resize((224, 224))  # Resize image to 224x224
    image = tf.keras.preprocessing.image.img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = tf.cast(image, tf.float32)  # Ensure image is float32 type
    image /= 255.0  # Rescale pixel values to [0, 1] just like in the training phase
    return image


def predict(image):
    # Run model prediction on the image
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.title('Skin Lesion Classification App')
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True)

# Use a container to hold the images and predictions to refresh them when new files are uploaded
container = st.container()

if uploaded_files:
    # Clear the container for new uploaded files
    container.empty()
    # Create a new container for the new images and predictions
    with container:
        # Display images in a grid (3 columns in this example)
        cols_per_row = 3
        cols = st.columns(cols_per_row)
        for idx, uploaded_file in enumerate(uploaded_files):
            # Open image
            bytes_data = uploaded_file.read()
            image = Image.open(io.BytesIO(bytes_data))
            # Preprocess and predict
            with st.spinner(f'Classifying image {idx + 1}...'):
                prediction = predict(image)
            # Display image in the appropriate column
            col = cols[idx % cols_per_row]
            col.image(image, use_column_width=True, caption=f'Uploaded Image {idx + 1}')
            # Display prediction in the appropriate column
            confidence = prediction[0][0] * 100
            if prediction[0] < 0.5:
                confidence = 100 - confidence
                col.success(f'Benign: {confidence:.2f}% confidence ✅')
            else:
                col.error(f'Malignant: {confidence:.2f}% confidence ❌')
# Run this with `streamlit run your_script.py`
