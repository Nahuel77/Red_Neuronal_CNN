# Convolutional Neural Network (CNN) that detects handwritten digits

Previously, I created an MLP network that performed the same task.

Now I am moving from MLP to CNN with the goal of improving accuracy.

## Typical CNN Architecture

Input (image)  
↓  
Convolution + ReLU  
↓  
Pooling  
↓  
Convolution + ReLU  
↓  
Pooling  
↓  
Flatten  
↓  
Fully Connected  
↓  
Softmax (classification)

### Just as in MLP we initialized weights randomly and adjusted them using backpropagation, here what will be initialized and trained are the *convolution kernels*\*.

![alt text](miscellaneous/image.png)

Although the formula may look complex, convolutions can be explained simply for the purpose of this example.

We have an image that becomes a value map between 0 and 1:

    x_train = train.drop(columns=['label']).values / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)


As in MLP, I remove the label column, normalize the values, and reshape them into 28×28 matrices. The convolution operates on this map.

Below is a simple convolution example using a 3×3 matrix and a 2×2 kernel:

I = input image  
K = kernel/filter  
Y = feature map

**Interpretation:**  
— Slide the kernel across the image  
— Multiply each kernel element by the corresponding pixel  
— Sum everything → one value in the feature map

I = [[1, 2, 0],  
     [0, 1, 3],  
     [1, 2, 2]]

K = [[1, 0],  
     [0, -1]]

First step (top left):  
1×1 + 2×0 + 0×0 + 1×(-1) = 0  
Second step (top right):  
2×1 + 0×0 + 1×0 + 3×(-1) = -1  
Third step (bottom left):  
0×1 + 1×0 + 1×0 + 2×(-1) = -2  
Fourth step (bottom right):  
1×1 + 3×0 + 2×0 + 2×(-1) = -1

Y = [[0, -1],  
     [-2, -1]]

In our algorithm, K is initialized randomly and adjusted through backpropagation.

The model uses:

    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),


TensorFlow applies 32 kernels of size 3×3 to the 28×28 image, producing 26×26 outputs.

Then we apply MaxPooling:

    MaxPooling2D((2,2))


Pooling 2×2 over a 26×26 matrix yields a 13×13 output.

Example:

[[1, 3, 2, 4],  
 [5, 6, 1, 2],  
 [0, 2, 3, 1],  
 [1, 0, 2, 4]]

2×2 Pooling:

[[6, 4],  
 [2, 4]]

From the 32 resulting maps, we convolve again with 63 kernels 3×3 and apply another 2×2 pooling, obtaining 63 maps of size 5×5.

    Conv2D(63, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),


You can think of this as a single 63×5×5 volume (1575 values).

`Flatten()` converts these 1575 values into a single vector, and `Dense(128, relu)` connects them to 128 neurons.

A major difference between MLP and CNN is that CNNs try to interpret *visual structures* (curves, straight lines, circles), not just statistical patterns.

To prevent overfitting, we use Dropout:

    Dropout(0.5)


This randomly disables half of the 128 neurons, forcing the network not to rely too heavily on specific visual patterns.

The model achieved **99.19% accuracy**.

Total params: 221,545 (865.41 KB)  
Trainable params: 221,545  
Non-trainable params: 0

Training results:

Epoch 1/10 — accuracy: 0.9187 — val_accuracy: 0.9785  
Epoch 2/10 — accuracy: 0.9728 — val_accuracy: 0.9854  
Epoch 3/10 — accuracy: 0.9803 — val_accuracy: 0.9864  
Epoch 4/10 — accuracy: 0.9827 — val_accuracy: 0.9892  
Epoch 5/10 — accuracy: 0.9860 — val_accuracy: 0.9886  
Epoch 6/10 — accuracy: 0.9881 — val_accuracy: 0.9907  
Epoch 7/10 — accuracy: 0.9883 — val_accuracy: 0.9899  
Epoch 8/10 — accuracy: 0.9897 — val_accuracy: 0.9868  
Epoch 9/10 — accuracy: 0.9914 — val_accuracy: 0.9908  
Epoch 10/10 — accuracy: 0.9920 — val_accuracy: 0.9919  

\* https://en.wikipedia.org/wiki/Convolution  
\*\* See the MLP README.md (https://github.com/Nahuel77/Red_Neuronal_MLP)
