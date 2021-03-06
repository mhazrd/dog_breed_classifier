# Dog Breed Classifier 

### Project Overview: 

It's not easy to differentiate multiple dog breeds, but with an advancement of machine learning, deep learning algorithm can determine dog breeds better than human. The high accuracy will make the model be used in a mobile app to help users to know what dog breeds he is looking at. Given multiple images including multiple dog breeds, the deep learning model will be trained to classify their breeds. This will be a example of image classification task. In this project, images of different dog breeds are used to train Convolutional neural network model

### Problem Statement:

Machine Learning model is trained on the dog breed image dataset. Convolutional Neural network architecture is selected because it's proven to be good at image tasks. The NN model is optimized to minimize cross entropy loss which is a proxy metric for accuracy.

### Metrics

The main task to solve is to classify dog-image into its breeds. There are 133 breeds and there is no high skew on a particular breed. Therefore accuracy will be a good metric to assess the ML model's performance on the classification task. If the dataset labels are highly skewed, the other metrics like f1-score could be used.

The accuracy is good for the model evaluation, but the deep learning model requires a continuous metric which the algorithm can optimize for. Cross-entropy loss is a continuous metric and can be used as a proxy for accuracy because it is an upper bound on the accuracy metric 

- Evaluation metric: Accuracy on dog-breed labels
- Optimization metric: Cross entropy on dog-breed labels


### Data Exploration

#### Dataset:

##### Dog images: 

- Total images: 8351 images
  - Training: 6680 images
  - Validation: 835 images
  - Test: 836 images
- Total Categories: 133 dog breeds
  - Smallest number of label: ('Norwegian_buhund', 26 images) 
  - Largest number of label: ('Alaskan_malamute', 77 images)
- Image channels: 3 channel (RGB)
- Image sizes: Variation of sizes
  - Smallest size: 112 x 120
  - Largest size: 4278 x 3744 

##### Label distribution:

![](images/distribution.png)


### Methodology

#### Data preprocessing

The CNN expects the input image to be of a specific size. But the dog image dataset has a variety of image sizes. Therefore these images are resized to 224 x 224 before it's fed into the CNN model.
The usual image RGB pixel values are encoded 3 channel with 8 bit values ranging from 0 to 255. For better model training, these values are usually normalised and the process can depend on the dataset and model architecture. In the final model, VGG19 back-bone model is used to extract features from images, therefore VGG19 normalization process is applied to the dog-breed image dataset.

#### Implementation

- Metrics
    - Evaluation metric: Accuracy on dog-breed labels
    - Optimization metric: Cross entropy on dog-breed labels
- Algorithm
    - Model: Convolutional Neural Network (VGG19 + classification layers)
    - Optimization algorithm: Adam
    - As the optimization process can make the model overfit to the training dataset. This can make it tricky to find an optimal model during training is in progress. Keras framework supports a convenient callback `ModelCheckpoint` which can track the metric as training progresses and stores the model state at an epoch where the model is optimal regarding the metric

#### Overall Steps 

1. The images are preprocessed as follows:
    1. The image is resized to 224 x 224
    2. The image transformed into 3-D tensors
    3. The image pixel values are normalised to conform to the backbone model's expectation
2. The backbone model (eg. VGG16/19) extracts features from the preprocessed image
3. The features are fed into the classification layer which is fine-tuned for dog-breed classification task
4. Final model is created as a composition of the backbone-model and the classification layer

#### Refinement

Initially, a fully custom CNN model is used. But it was not easy to find a combination showing a good performance because of the large search space regarding hyperparameters, model architecture and optimization algorithms. This fully custom CNN model achieved ~2% accuracy which is still better than random guessing but far from a usable performance.
In order to reduce the search space, previously proven model architecture is used to extract features and it's customized with final few layers for the custom task. It basically uses a transfer learning from a more general image task into dog-breed classification task, reduces the search space of model architecture into a few final layers and helped improve the accuracy performance.


### Results

#### Final Model
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


#### Model Validation

- Accuracy on test dataset: 75.48%
- Generalization Test
   - Even though the model worked well on the unseen test data. This test data comes from same dataset (with random sampling) and the general obtainable dog image might not share the charcteristic with the curated dataset. Therefore, a few dog images were retrieved from the web and used to check if the model can actullay generalize beyond the curated dataset.
   - These samples images from the web are stored in test_img/ and used to perform a simple final check on model's generalizability
- Robustness Test
    - The model might be sensitive to each training process or a little difference on the training dataset. Therefore, it's useful to check if the test performance is achievable regardless of this kind of randomness
    - To check model's robustness, the training dataset is split into 5 folds randomly, and the model is trained and assessed on the test dataset 5 times with each fold removed. The test accuracy results are as follows:
        - (74.16%, 71.77%, 73.68%, 75.60%, 74.40%)
    - The mean is 73.92% and standard deviation is 1.24%. Therefore, it can be safely assumed that the model is robust against a small random perturbation and can achieve >= 70% accuracy

#### Justification

As explored inside the jupyter notebook, It's not easy to get a good performance with the complete custom model. Transfer learning really worked well and helped improve the model performance a lot. This improvement comes from good generalizable features. In case of fully custom CNN model, image features and labels need to be learned together on the dog-breed dataset. However, the backbone-model like VGG19 is trained on much larger dataset (ImageNet) to learn generalizable good features and only the final classification layer is learned to use these features to classify dog breeds. Therefore, the learning task becomes much easier and helps the model learn well on the given labels.


### Conclusion

#### Reflection
Dog breed classification is not an easy task for human because there are a lot of breeds and sometimes the difference can be subtle to recognize. Trained ML model can help human know what breed he is looking at, because the model can learn the subtle differences and successfully classify dog breeds. In this project, CNN model and transfer learning technique are used to achieve a high accuracy on the task. The final solution is made with pretrained VGG19 backbone network and fine-tuned classification layers on dog-breed labels. A few difficulties were:

1. Huge search space in learning CNN model due to a lot of levers to tune (eg. architecture, hyperparameters, optimization algorithms)
2. Uncertainty on the data size if it's big enough to learn good visual features and dog-breed labels together

It had too many levers to consider/search when developing a fully custom model. But with transfer-learning technique, it became much easier to get a good performance on custom task because these models were already trained and tuned carefully to extract good visual features which can generalize across a lot of image domain tasks.

#### Potential improvements

The fine-tuned classification layer achieves a much better accuracy than the custom model without a pretrained backbone. However, It wasn't be easy to achieve an accuracy over 80% with just VGG19 + Dense-layer based architecture. It can be further optimized to improve with the following methods:

1. Use more sophisticated optimization algorithm with proper hyperparameters
2. Search a better architecture for the classification layer
3. Try using more than 1 backbone pre-trained architecture

The option 3 will make the overall model more complex with more parameters than the current solution because each pre-trained NN already has a lot of parameters. Depending on a hardware spec, it might require a distributed training to train the model as it can require more memory to store model weights and data together.


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
- README.md: Writeup of the project


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