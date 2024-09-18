import segmentation_models as sm
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from typing import Tuple
from utils.visualization import visualize_results

# Set the framework to TensorFlow
sm.set_framework('tf.keras')

# Set some parameters
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Load your data
def load_data(data_dir, resize_dim: Tuple[int, int] = (384, 384)):
    img_dir = os.path.join(data_dir, 'test_img')
    lab_dir = os.path.join(data_dir, 'test_lab')
    
    images = []
    labels = []
    
    for img_name in os.listdir(img_dir):
        # Load image
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, resize_dim)  # Resize to a fixed size
        images.append(img)
        
        # Load corresponding label
        lab_name = os.path.splitext(img_name)[0] + '.png'
        lab_path = os.path.join(lab_dir, lab_name)
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        lab = cv2.resize(lab, resize_dim, interpolation=cv2.INTER_NEAREST)  # Resize labels
        labels.append(lab)
    
    return np.array(images), np.array(labels)

x_test, y_test = load_data('./data/')
x_test = preprocess_input(x_test)


# Before evaluating
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

# Print shapes to verify
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# Before the evaluation loop
y_test = (y_test > 0).astype('float32')

# All models

models = {
    'Unet': sm.Unet,
    'Linknet': sm.Linknet,
    'PSPNet': sm.PSPNet,
    'FPN': sm.FPN
}

for name, sm_model in models.items():
    print(f"Testing {name}...")
    # Define the base model
    base_model = sm_model(BACKBONE, classes=1, activation='sigmoid', encoder_weights='imagenet')

    # Add a Lambda layer to process the output
    inputs = base_model.input
    x = base_model(inputs)
    if name == 'PSPNet':
        outputs = tf.keras.layers.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), tf.float32))(x)
    else:
        outputs = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(x)

    # Create a new model with the added layer
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    # Evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=1)
    print(f"{name} - Loss: {scores[0]:.5f}, IoU Score: {scores[1]:.5f}")

    # Generate predictions
    predictions = model.predict(x_test)

    # Visualize results for the first 5 images
    for i in range(5):
        original_image = x_test[i]
        true_mask = y_test[i]
        pred_mask = predictions[i]

        # Denormalize the image
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

        visualize_results(original_image, true_mask, pred_mask, name, i)

print("Visualization complete. Results saved in ./results directory.")