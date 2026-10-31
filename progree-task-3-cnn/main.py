
import os
import json
import random
from pathlib import Path

# --------------------------------------------------
# 1. Project paths and reproducibility
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
KERAS_DIR = PROJECT_DIR / ".keras"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KERAS_DIR.mkdir(parents=True, exist_ok=True)

# Store the CIFAR-10 dataset cache on E: drive.
os.environ["KERAS_HOME"] = str(KERAS_DIR)

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Training configuration
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

print("=" * 60)
print("PROGREE INTERNSHIP - TASK 3")
print("CIFAR-10 Multi-Class Image Classification using CNN")
print("=" * 60)

print(f"TensorFlow version: {tf.__version__}")
print(f"Project directory: {PROJECT_DIR}")
print(f"Dataset cache: {KERAS_DIR}")
print(f"Outputs directory: {OUTPUT_DIR}")


# --------------------------------------------------
# 2. Load CIFAR-10 dataset
# --------------------------------------------------

print("\n[1/6] Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print(f"Training images: {x_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Testing images: {x_test.shape}")
print(f"Testing labels: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")


# --------------------------------------------------
# 3. Build data augmentation layers
# --------------------------------------------------

print("\n[2/6] Configuring data augmentation...")

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
        ),
    ],
    name="data_augmentation",
)


# --------------------------------------------------
# 4. Build custom CNN architecture
# --------------------------------------------------

print("\n[3/6] Building the CNN model...")

inputs = tf.keras.Input(shape=(32, 32, 3), name="input_image")

# Augmentation is active during training only.
x = data_augmentation(inputs)

# Convert pixel values from 0-255 to 0-1.
x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

# Convolution block 1
x = tf.keras.layers.Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    name="conv2d_1",
)(x)
x = tf.keras.layers.Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    name="conv2d_2",
)(x)
x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    name="max_pooling_1",
)(x)
x = tf.keras.layers.Dropout(0.25)(x)

# Convolution block 2
x = tf.keras.layers.Conv2D(
    64,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    name="conv2d_3",
)(x)
x = tf.keras.layers.Conv2D(
    64,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    name="conv2d_4",
)(x)
x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    name="max_pooling_2",
)(x)
x = tf.keras.layers.Dropout(0.30)(x)

# Convolution block 3
x = tf.keras.layers.Conv2D(
    128,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    name="conv2d_5",
)(x)
x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    name="max_pooling_3",
)(x)
x = tf.keras.layers.Dropout(0.30)(x)

# Fully connected classification head
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="fully_connected",
)(x)
x = tf.keras.layers.Dropout(0.40)(x)

outputs = tf.keras.layers.Dense(
    10,
    activation="softmax",
    name="class_probabilities",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="CIFAR10_CNN",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

print("\n[4/6] Training the CNN model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(OUTPUT_DIR / "training_history.csv")
    ),
]

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)


# --------------------------------------------------
# 6. Evaluate on unseen test images
# --------------------------------------------------

print("\n[5/6] Evaluating the model on test data...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1,
)

best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

metrics = {
    "dataset": "CIFAR-10",
    "num_classes": len(class_names),
    "class_names": class_names,
    "training_images": int(len(x_train)),
    "testing_images": int(len(x_test)),
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history.history["loss"]),
    "best_validation_accuracy": float(
        max(history.history["val_accuracy"])
    ),
    "best_validation_loss": float(
        min(history.history["val_loss"])
    ),
    "best_validation_accuracy_epoch": best_epoch,
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "random_seed": SEED,
}

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# Save the trained model.
model.save(OUTPUT_DIR / "cifar10_cnn.keras")


# --------------------------------------------------
# 7. Plot training and validation accuracy
# --------------------------------------------------

print("\n[6/6] Creating graphs and saving results...")

epochs_range = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(9, 5))
plt.plot(
    epochs_range,
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy",
)
plt.plot(
    epochs_range,
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy",
)
plt.title("CIFAR-10 CNN: Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_validation_accuracy.png",
    dpi=150,
)
plt.close()

# Plot training and validation loss.
plt.figure(figsize=(9, 5))
plt.plot(
    epochs_range,
    history.history["loss"],
    marker="o",
    label="Training Loss",
)
plt.plot(
    epochs_range,
    history.history["val_loss"],
    marker="o",
    label="Validation Loss",
)
plt.title("CIFAR-10 CNN: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_validation_loss.png",
    dpi=150,
)
plt.close()

# Generate a small sample of test predictions.
sample_images = x_test[:12]
sample_labels = y_test[:12].flatten()

probabilities = model.predict(
    sample_images,
    batch_size=BATCH_SIZE,
    verbose=0,
)
predicted_labels = np.argmax(probabilities, axis=1)

prediction_results = []

for index in range(len(sample_images)):
    prediction_results.append(
        {
            "image_number": index + 1,
            "actual_class": class_names[int(sample_labels[index])],
            "predicted_class": class_names[int(predicted_labels[index])],
            "confidence": float(
                np.max(probabilities[index])
            ),
            "correct": bool(
                sample_labels[index] == predicted_labels[index]
            ),
        }
    )

with open(
    OUTPUT_DIR / "sample_predictions.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(prediction_results, file, indent=4)

print("\n" + "=" * 60)
print("TASK 3 COMPLETED")
print("=" * 60)
print(f"Best validation accuracy: {metrics['best_validation_accuracy']:.4f}")
print(f"Test accuracy:            {test_accuracy:.4f}")
print(f"Test loss:                {test_loss:.4f}")
print(f"Epochs completed:         {metrics['epochs_completed']}")
print(f"Best validation epoch:    {best_epoch}")
print(f"\nAll results saved in: {OUTPUT_DIR}")
print(" - training_validation_accuracy.png")
print(" - training_validation_loss.png")
print(" - training_history.csv")
print(" - metrics.json")
print(" - sample_predictions.json")
print(" - cifar10_cnn.keras")
