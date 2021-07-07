import os
import io
import numpy as np
from flask import Flask, request, jsonify

from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from PIL import Image

from labels import LABELS

app = Flask(__name__)

# Fine-tuned classification layer
classification_layer = Sequential([
    GlobalAveragePooling2D(input_shape=(7, 7, 512)),
    Dense(256, activation='relu'),
    Dense(133, activation='softmax')
])
classification_layer.load_weights('ws.VGG19.hdf5')

def extract_VGG19(tensor):
    '''
    It extracts features using VGG19 model from given image

    :param tensor: The image tensor
    :returns a feature tensor
    '''
	from keras.applications.vgg19 import VGG19, preprocess_input
	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def classify(features):
    '''
    It determines the dog breed given a feature tensor extracted from a dog image

    :param features: The feature tensor
    :returns a label string (one of 133 dog breeds)
    '''
    predicted_vector = classification_layer.predict(features)

    return LABELS[np.argmax(predicted_vector)]

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Web API to accept a dog image and return its breed label in JSON

    :returns a dog-breed label string wrapped in JSON to a client
    '''
    if request.method == 'POST':
        img = request.files.get('image')
        if img:
            img_data = Image.open(io.BytesIO(img.read()))
            img_arr = image.img_to_array(img_data.resize((224, 224)))
            img_arr = np.expand_dims(img_arr, axis=0)

            features = extract_VGG19(img_arr)
            label = classify(features)

            return jsonify({'dog_breed': label})


if __name__ == '__main__':
    app.run()