# Sign-Language-Recognition-Neural-Network

#### Project by 
  #### &nbsp;&nbsp;&nbsp;&nbsp; - Swapnanil Ray *(Institute Of Engineering & Management, Kolkata)* 
  #### &nbsp;&nbsp;&nbsp;&nbsp; - Anjali Shaw *(Institute Of Engineering & Management, Kolkata)*
  
#### Supervised and Guided by 
  #### &nbsp;&nbsp;&nbsp;&nbsp; - Ushnish Sarkar *(Variable Energy Cyclotron Centre, Kolkata)*
  
This project deals with the Finger Spelling Kinect 2011 Dataset. Which contains pictures of the standard American Sign Language depicting the English Alphabets.
[Link to the Dataset](http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2)

The Dataset structure is as follows  

&nbsp;&nbsp;&nbsp;&nbsp; - 5 Main Directories (A,B,C,D,E) 

&nbsp;&nbsp;&nbsp;&nbsp; - Each Main Directory contains 24 Sub Directories symbolising the 24 Letters(The letters J and Z are excluded from the dataset since they are expressed as moving gestures and not static gestures.) of the English Alphabet.

To Unzip the data use :
```
tar -xvf /path/to/fingerspelling5.tar.bz2
```


# Introduction

### Defining Neural Networks -

Neural networks reflect the behavior of the human brain, allowing computer programs to recognize patterns and solve common problems in the fields of AI, machine learning, and deep learning.

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

[*Fig: Figurative Representation of Neural Network (Source:* [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks))](https://lh4.googleusercontent.com/Ko0kFJ-GStX6Ase_lYQO29xH3BH4LUXo8_OMRj4fl68T_vMR_QUGAMe5MgFYBL68M2nfEJROmAeQgXBak9a9wxXEXW32qJdDDzNfdWJPAIf_UKrLVmY330xoPdRsuH5WabvVP7lW8_FVJoYLDA)

*Fig: Figurative Representation of Neural Network (Source:* [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks))

### Working

Artificial neural networks (ANNs) are comprised of a node layer, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

Thinking of each individual node as its own linear regression model, composed of input data, weights, a bias (or threshold), and an output. The formula would look something like this:

[*Fig: Weights and Bias (Source:* [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks))](https://lh4.googleusercontent.com/AMmKrME2pASbcqUuf-QILp54PhHTPWQKsqdUyWTCG2vv2U_Q2KqF9MEidxBEpNTjDKKQcMg0hmkfXVKWzlLUwr5KhdmJv848EYMjVBPv_ORHp6D68yn8m8VtYDjhe_oMGHoHWI1LtuDAQRpXHw)

*Fig: Weights and Bias (Source:* [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks))

[*Fig: Output Function (Source:* [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks))](https://lh5.googleusercontent.com/wGBYh2HrVbeO-RCeUusy25g0xU8xg-1y6KXAnzNrnWMHASfXljofgWiMD7TsFXRqtiiQ2C8ngz53QZJwe6qlhBfeLNYTJ4MtU35rPRjGqAsfctZKYZTNyRklL5OnIPQMZ_dN6qSRP0MHmg9CIA)

*Fig: Output Function (Source:* [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks))

  Neural Networks is the essence of Deep Learning.



# CNN & a Brief about LNN

### Defining LNN:

Before diving into Convolutional Neural Networks, let us first define what are Linear & Non-Linear Neural Networks.

- The neural network without any activation function in any of its layers is called a linear neural network.
- The neural network which has action functions like ReLU, sigmoid or tanh in any of its layer or even in more than one layer is called non-linear neural network.

### Applying LNN for Image Processing :

We used the Fashion MNIST Dataset for Image Processing using Linear Neural Networks. 

The Model Summary is as such:

![*Fig: Model Summary*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605542/Untitled_i2j3dk.jpg)

*Fig: Model Summary*

The above model was trained across 15 Epochs with a batch size of 1875

![*Fig: Training of the Model through the Epochs*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605540/Untitled_1_epgc02.jpg)

*Fig: Training of the Model through the Epochs*

The Model yields the following results:

![*Fig: Test Accuracy*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605540/Untitled_2_a7gfhw.jpg)

*Fig: Test Accuracy*

### Drawbacks of using LNN:

While solving an image classification problem using LNN, the first step is to convert a 2-dimensional image into a 1-dimensional vector (flattening) prior to training the model. This has two drawbacks:

- The number of trainable parameters increases drastically with an increase in the size of the image
- LNN loses the spatial features of an image. Spatial features refer to the arrangement of the pixels in an image.

### Defining CNN

A convolutional neural network, or CNN, is a deep learning neural network designed for processing structured arrays of data such as images. Convolutional neural networks are widely used in computer vision and have become the state of the art for many visual applications such as image classification, and have also found success in natural language processing for text classification.

Convolutional neural networks are distinguished from other neural networks by their superior performance with image, speech, or audio signal inputs. They have three main types of layers, which are:

- Convolutional layer
- Pooling layer
- Fully-connected (FC) layer

The convolutional layer is the first layer of a convolutional network. While convolutional layers can be followed by additional convolutional layers or pooling layers, the fully-connected layer is the final layer. With each layer, the CNN increases in its complexity, identifying greater portions of the image. Earlier layers focus on simple features, such as colors and edges. As the image data progresses through the layers of the CNN, it starts to recognize larger elements or shapes of the object until it finally identifies the intended object.

- Convolutional Layer
    - The convolutional layer is the core building block of a CNN, and it is where the majority of computation occurs. It requires a few components, which are input data, a filter, and a feature map. Let’s assume that the input will be a color image, which is made up of a matrix of pixels in 3D. This means that the input will have three dimensions—a height, width, and depth—which correspond to RGB in an image. We also have a feature detector, also known as a kernel or a filter, which will move across the receptive fields of the image, checking if the feature is present. This process is known as a convolution.
    
    ![*Fig: Convolutional Layer*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_3_qd4jce.jpg)
    
    *Fig: Convolutional Layer*
    
- Pooling Layer
    - Pooling layers, also known as downsampling, conducts dimensionality reduction, reducing the number of parameters in the input. Similar to the convolutional layer, the pooling operation sweeps a filter across the entire input, but the difference is that this filter does not have any weights. Instead, the kernel applies an aggregation function to the values within the receptive field, populating the output array. There are two main types of pooling:
        - **Max pooling:** As the filter moves across the input, it selects the pixel with the maximum value to send to the output array. As an aside, this approach tends to be used more often compared to average pooling.
        - **Average pooling:** As the filter moves across the input, it calculates the average value within the receptive field to send to the output array.
    
    While a lot of information is lost in the pooling layer, it also has a number of benefits to the CNN. They help to reduce complexity, improve efficiency, and limit risk of overfitting.
    
- Fully Connected Layer
    - The name of the full-connected layer aptly describes itself. As mentioned earlier, the pixel values of the input image are not directly connected to the output layer in partially connected layers. However, in the fully-connected layer, each node in the output layer connects directly to a node in the previous layer.
    - This layer performs the task of classification based on the features extracted through the previous layers and their different filters. While convolutional and pooling layers tend to use ReLu functions, FC layers usually leverage a softmax activation function to classify inputs appropriately, producing a probability from 0 to 1.
    
    ### Applying CNN for Image Processing
    
    We used the Fashion MNIST Dataset for Image Processing using Convolutional Neural Networks. 
    
    The Model Summary is as such:
    
    ![*Fig: Model Summary*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605540/Untitled_4_lemgvi.jpg)
    
    *Fig: Model Summary*
    
    The model is trained across 90 Epochs and a Batch Size of 938:
    
    ![*Fig: Training of the Model through the Epochs*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605540/Untitled_5_tkkttl.jpg)
    
    *Fig: Training of the Model through the Epochs*
    
    The Model yields the following results:
    
    ![*Fig: Test Accuracy*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605540/Untitled_6_mkzo3r.jpg)
    
    *Fig: Test Accuracy*



# Brief about Sign Languages and the ASL Finger Spelling Kinect 2011 Dataset

### Definition:

Sign languages (also known as signed languages) are languages that use the visual-manual modality to convey meaning. Sign languages are expressed through manual articulations in combination with non-manual elements. Sign languages are full-fledged natural languages with their own grammar and lexicon. Sign languages are not universal and are usually not mutually intelligible, although there are also similarities among different sign languages.

For this project we will be specifically working on the American Sign Language commonly known as the ASL.

American Sign Language (ASL) is a natural language that serves as the predominant sign language of Deaf communities in the United States and most of Anglophone Canada. ASL is a complete and organized visual language that is expressed by both manual and nonmanual features. Besides North America, dialects of ASL and ASL-based creoles are used in many countries around the world, including much of West Africa and parts of Southeast Asia. ASL is also widely learned as a second language, serving as a lingua franca. ASL is most closely related to French Sign Language (LSF). It has been proposed that ASL is a creole language of LSF, although ASL shows features atypical of creole languages, such as agglutinative morphology.

![*Fig: The American Sign Language (Source:*[https://en.wikipedia.org/wiki/American_Sign_Language](https://en.wikipedia.org/wiki/American_Sign_Language))](https://res.cloudinary.com/redhatpanda/image/upload/v1657605540/Untitled_7_ugfxyl.jpg)

*Fig: The American Sign Language (Source:*[https://en.wikipedia.org/wiki/American_Sign_Language](https://en.wikipedia.org/wiki/American_Sign_Language))

The ASL for the English Alphabet can be depicted using the following symbols shown in the figure below:

![*Fig: The American Sign Language Sheet* ](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_8_prt044.jpg)

*Fig: The American Sign Language Sheet* 

For this Project we’ll be working on the American Sign Language Finger Spelling Kinect 2011 Dataset which is publicly available and free to download. 

Link to the dataset: [http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2](http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2)

This Dataset has the following features :

| Classes | Subjects | Samples | Data | Type |
| --- | --- | --- | --- | --- |
| 24 | 5 | 131000 | 2.1GB | Images (Depth+RGB) |

The Directory structure of the dataset is as follows:

- There are 5 Subject directories A,B,C,D,E
    - Each Subject has 24 sub-directories that denote the letters of the English Alphabet (a-z).
        - Each Letter directory contains images that are distributed across Depth & RGB Versions.
        

**Note**: The letters ‘J’ and ‘Z’ are excluded from the dataset because they are depicted using a set of motions and aren’t static in nature.



# Data Preprocessing

The data presented in the ASLFSK2011 Dataset is nothing but images. We know that, we can feed an image directly to a Neural Network.

We need to convert the image into a set of numbers, so that the NN can actually work on the numerical data, extract the essential features and producing accurate predictions.

For this purpose we needed Data Preprocessing on our Dataset.

We first begin by scaling down the dataset to a more usable number. Since the dataset has 131000 images, it is essentially very difficult to process all the images.

As mentioned in the previous section regarding the directory structure of the Dataset, we select 64 images (we have used only the Color versions of the images and not the Depth versions) from each of the letter for each subject. 

Hence we have a total of *64 images * 24 letters * 5 subjects = **7680** Images.*

We used Mediapipe to parse through all the Images and annotate them with the 21 Hand Landmarks as described in the figure below :

![*Fig: The 21 Hand Landmarks (Source:* [https://google.github.io/mediapipe/solutions/hands.html](https://google.github.io/mediapipe/solutions/hands.html))](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_9_kuqjup.jpg)

*Fig: The 21 Hand Landmarks (Source:* [https://google.github.io/mediapipe/solutions/hands.html](https://google.github.io/mediapipe/solutions/hands.html))

We save the x, y & z coordinates of the Hand Landmarks for the all the images on which mediapipe detects a hand.

We now find the adjacent nodes, while referring the figure shown above. We find that there are 26 unique pairs of adjacent nodes as shown below. 

![*Fig: Unique of Adjacent Nodes*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_10_zobtau.jpg)

*Fig: Unique of Adjacent Nodes*

We find the angle between such nodes for every image in our dataset and save that in a csv file. This is the final data that we use to feed into our model.

The spatial co-ordinates of the particular hand will change with the position of hand in the image frame and hence the model will not be able to extract the essential features and be able to predict accurately. Hence we chose the angles between the adjacent nodes as our primary feature, since the sign language for a particular alphabet remains constant.


# Preparation & Evaluation of the Model

After further processing of the data, we then define go on and feed the data to our model.

Before feeding the data into the model, we split the data into Training & Testing Sets.

Since our data is already in a flattened form, we will use Linear Neural Network, and not CNN. 

We define our model as shown in the model summary below:

![*Fig: Model Summary*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_11_awnufy.jpg)

*Fig: Model Summary*

We train & fit our model across 20 Epochs with a Batch Size of 159:

![*Fig:* Training of the Model through the Epochs](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_12_lpdkzr.jpg)

*Fig:* Training of the Model through the Epochs

We have tried to plot the model accuracy and the model loss across the different epochs which can be visualised below:

![*Fig:* Model Accuracy vs Epoch](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_13_iyljxz.jpg)

*Fig:* Model Accuracy vs Epoch

![*Fig:* Model Loss vs Epoch](https://res.cloudinary.com/redhatpanda/image/upload/v1657605541/Untitled_14_h2tjpm.jpg)

*Fig:* Model Loss vs Epoch

The model evaluation gives the following results:

![*Fig: Test Accuracy*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605542/Untitled_15_setuy1.jpg)

*Fig: Test Accuracy*

The predicted values are manually checked against the Test Label values and the following results were observed:

![*Fig: Test Label Values vs Predicted Values*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605542/Untitled_16_nm39af.jpg)

*Fig: Test Label Values vs Predicted Values*

In the figure, we can see in the Left Column there are the Test Label Values and in the Right Column there are the Predicted Values. The number in each column in actually the index of the alphabet (0-based Indexing, i.e. a=0, b=1, … , z = 25)

We also find that not all values are matching when compared. Hence we found the number and the percentage of misclassified predictions, the results are below.

![*Fig: Count of misclassified values*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605542/Untitled_17_qbupon.jpg)

*Fig: Count of misclassified values*

![*Fig: Percentage of misclassified values*](https://res.cloudinary.com/redhatpanda/image/upload/v1657605542/Untitled_18_t3zfg1.jpg)

*Fig: Percentage of misclassified values*


# Conclusion

We implemented our model using Linear Neural Network, since the data we possessed was already flattened, hence not using CNN.

We were able to detect the Finger Spelling only for 24 letters. The letter **J** and **Z** are excluded intentionally as they are in-motion gestures. 

To detect the excluded letters, we have to use much more complicated architectures like RNN and LSTM, which is beyond the scope of this work. 

Link to Code Repository: [https://github.com/swapanjaliVeccProject/Sign-Language-Recognition-Neural-Network](https://github.com/swapanjaliVeccProject/Sign-Language-Recognition-Neural-Network)

# References

[1] [https://google.github.io/mediapipe/](https://google.github.io/mediapipe/)

[2] [https://www.ibm.com/in-en/cloud/learn/neural-networks](https://www.ibm.com/in-en/cloud/learn/neural-networks)

[3] [http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2](http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2)

[4] [https://www.tensorflow.org/tutorials/keras/classification](https://www.tensorflow.org/tutorials/keras/classification)
