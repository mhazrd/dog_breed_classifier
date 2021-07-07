# Dog Breed Classifier 

### About: 

It's not easy to differentiate multiple dog breeds, but with an advancement of machine learning, deep learning algorithm can determine dog breeds better than human. Given multiple images including multiple dog breeds, the deep learning model will be trained to classify its breeds. This will be a example of image classification task.

### Model:
The dataset includes 133 breeds of dog images and is used to train Convolutional Neural Netowrk model. The model is trained with cross entropy loss and evaluated with its accuracy on test dataset. The high accuracy will make the model be used in a mobile app to help users to know what dog breeds he is looking at.

### Dataset:

Dog images: 
- Totla images: 8351 (133 categorical labels)
  - Training: 6680
  - Validation: 835
  - Test: 836

- Image size: 226 x 226
- Image channels: 3 channel (RGB)

### Procedures
1. The images are transformed into 3-D tensors and normalised to conform to the backbone model's expectation
2. The backbone model (eg. VGG16/19) extracts features from the image
3. The features are fed into the classification layer which is fine-tuned for dog-breed classification task
4. Final model is created as a composition of the backbone-model and the classification layer

### Final Model
- VGG19 model (Backbone without the fully-connected layers)
- Fine-tuned classification layers (# parameters: 165,509)
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_2 ( (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_4 (Dense)              (None, 133)               34181     
=================================================================
```

- Accuracy on test dataset: 75.48%

### Dicussion
As explored inside the jupyter notebook, It's not easy to get a good performance with the complete custom model. A few reasons could be:

1. There is a huge search space in architecture and training hyperparms to find an optimal combination
2. The given dataset might not be big enough to learn enough good visual features

Therefore, the backbone-model and transfer-learning can make it easier to get a good performance on custom task because these models are already trained carefully to extract good visual features with a lot of searches and data.

### Potential improvements

The fine-tuned classification layer achieves a much better accuracy than the custom model without a pretrained backbone. However, It wasn't easy to achieve an accuracy over 80% with just VGG19 + Dense-layer based architecture. It can be further optimized to improve with the following methods:

1. Use more sophisticated optimization algorithm with proper hyperparameters
2. Search a better architecture for the classification layer
3. Try using more than 1 backbone pre-trained architecture 

## How to test web app

- Go to app directory and start the Flask api server
```bash
cd app/
python app.py
```

- Go to test_img directory and try posting a sample image to the Flask api server
```bash
cd test_img/
curl -X POST -F image=@dog_boston-terrier2.jpg http://localhost:5000/predict
```


## File Structure

- haarcascades/ : This folder contains opencv-based human face detector
- images/ : This folder contains sample images
- saved_models/ : This folder contains fine-tuned classification layer trained weights
- test_img/ : This folder contains sample images to test the model
- app/ : This folder contains simple API web-app to classify dog images
- dog_app.ipynb: Main notebook for dog_breed classificer


## Dependencies

- python packages
  - opencv-python
  - h5py
  - matplotlib
  - numpy
  - scipy
  - tqdm
  - keras
  - scikit-learn
  - pillow
  - ipykernel
  - tensorflow
  - Flask
- Linux packages
  - curl