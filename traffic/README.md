# Traffic Sign Recognition Experimentation

During the experimentation with convolutional neural network architectures for traffic sign recognition on the GTSRB dataset, several configurations were tested:

1. **Initial Baseline**: The starting network utilized a single convolutional layer (32 filters, 3x3 kernel) followed by max-pooling (2x2) and a dense layer with 128 hidden units. While this converged reasonably well, it achieved only around 85-88% accuracy on the test set, showing signs of slight underfitting.

2. **Deeper Architecture**: Adding a second convolutional layer (64 filters, 3x3 kernel) with another max-pooling step significantly improved feature extraction, raising accuracy above 94%. Introducing a Dropout layer (rate 0.5) before the final output layer effectively controlled overfitting on training images, leading to better generalization on the unseen test partition with balanced loss.
