# MoleSafeScan App

Welcome to the MoleSafeScan App repository! This app is a tool for classifying skin lesions as benign or malignant using a convolutional neural network (CNN) based on the MobileNetV2 architecture. It is built with Streamlit for easy web deployment and TensorFlow for constructing and training the model.

## Disclaimer

This app is intended for educational and research purposes only and is not meant to be a substitute for professional medical advice or diagnosis.

## App Features

- Upload multiple images of skin lesions for classification.
- Classify lesions with a pre-trained MobileNetV2 model fine-tuned on a proprietary skin lesion dataset.
- View classification results directly in the app with confidence scores.

## Model Information

- **Architecture**: MobileNetV2 pre-trained on ImageNet.
- **Fine-tuning**: Started from layer 154 of MobileNetV2.
- **Test Accuracy**: 91.04%.
- **Dataset**: Proprietary dataset containing images of benign and malignant skin lesions.

## Requirements

To run the app, you'll need:
- Python 3.7+
- Streamlit
- TensorFlow 2.15.0
- NumPy
- Pillow (PIL)

## Local Setup

1. Clone the repository to your local machine.
2. Install the required Python packages:

    ```
    pip install streamlit tensorflow==2.15.0 numpy pillow
    ```

3. Run the app:

    ```
    streamlit run app.py
    ```

## Training the Model

The model was trained in a Jupyter notebook (`Skin_Cancer.ipynb`) with the following key steps:
- Data preprocessing and augmentation using `ImageDataGenerator`.
- Model definition with TensorFlow's Keras API.
- Fine-tuning on a skin lesion dataset from layer 154.
  

For detailed training procedures, hyperparameter tuning, and learning rate analysis, refer to the Jupyter notebook.

## Deployment

After training, the model is saved and can be converted to TensorFlow Lite format for deployment in mobile applications.
You can test the model here : https://psilly-billy-molesafescan.streamlit.app 

## Contributions

Contributions to the app or model are welcome. Please submit a pull request or open an issue if you have suggestions or find a bug.



## License

Distributed under the MIT License. See `LICENSE` for more information.
"""
