🔎 Model Analysis: Fast-EdgeLeaf TuneNet
1. Dataset

The dataset used is from Kaggle: Tomato Leaf Disease Dataset.

Images are categorized into multiple classes (healthy vs diseased leaves).

Images are resized to 224×224 pixels.

Data Augmentation is applied during training:

Rotation, zoom, shear, and horizontal flips.

Validation set is only rescaled (no augmentation).

2. Base Model

The architecture is built on MobileNetV3Small, imported from tensorflow.keras.applications.

Why MobileNetV3Small?

It is lightweight and optimized for edge devices (phones, IoT, low-power GPUs).

Uses depthwise separable convolutions for efficiency.

Suitable for fast inference in real-world agricultural settings.

3. Fast-EdgeLeaf TuneNet Model Structure

The notebook customizes MobileNetV3Small into a new tuned model:

Input Layer: 224×224×3 RGB image.

Feature Extractor:

MobileNetV3Small (pre-trained on ImageNet) without the top classification layer.

Extracts robust image features.

Global Average Pooling: Reduces feature maps into compact feature vectors.

Dense Layers:

Fully connected layer(s) for classification.

Dropout layers are added to reduce overfitting.

Output Layer:

Dense(num_classes, activation="softmax") for multi-class leaf disease classification.

4. Training Setup

Optimizer: Adam (commonly used for image classification).

Loss Function: Categorical Crossentropy (multi-class).

Metrics: Accuracy, Confusion Matrix, Classification Report.

Callbacks:

EarlyStopping → stops when validation loss stops improving.

ReduceLROnPlateau → reduces learning rate when training plateaus.

5. Inspiration Behind Fast-EdgeLeaf TuneNet

This model is inspired by efficient CNN architectures for mobile/edge deployment:

MobileNetV3

The backbone is MobileNetV3Small, which is optimized for speed and low power consumption.

Inspired by depthwise separable convolutions (from MobileNetV1) and SE-blocks (Squeeze-and-Excitation).

Transfer Learning

Pretrained ImageNet weights are reused.

Inspired by transfer learning approaches where large-scale pre-trained models are fine-tuned for smaller datasets.

Fine-tuning strategy

The notebook builds a TuneNet approach by adjusting final layers and hyperparameters to balance accuracy and speed.

So, Fast-EdgeLeaf TuneNet is essentially a fine-tuned MobileNetV3Small designed for fast and efficient leaf disease classification on edge devices.
