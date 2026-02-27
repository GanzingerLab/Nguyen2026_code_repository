#%%
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# =============================================================================
# Deep-learning classifier for synapse maturation (immature vs mature)
#
# Expected folder structure:
#   DATA_DIR/
#     immature/   (images)
#     mature/     (images)
#
# NOTE: This training script assumes each file corresponds to a single 2D image.
# If your dataset contains multi-page TIFF stacks, ensure they are exported as
# single-frame images (or modify the loader to read a specific frame/page).
# =============================================================================

# === Settings ===
DATA_DIR = r'C:\Users\castrolinares\Data analysis\SPIT_G\Random_forest_mature\V2\DL'
IMG_SIZE = (224, 224)       # resized to match MobileNetV2 input
BATCH_SIZE = 16
EPOCHS = 20                 # phase 1 (frozen backbone) maximum epochs
fine_tune_epochs = 30       # phase 2 (fine-tuning) maximum epochs
total_epochs = EPOCHS + fine_tune_epochs
CLASS_NAMES = ['immature', 'mature']

# === Load dataset ===
def load_dataset(data_dir, class_names):
    """Collect file paths and integer labels from DATA_DIR/class_name/."""
    image_paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.tif')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return image_paths, labels

# === Preprocess grayscale → RGB ===
def preprocess_grayscale_to_rgb(img):
    """
    Scale 12-bit grayscale images to [0, 1] and replicate to 3 channels.
    Replication is used because MobileNetV2 (ImageNet-pretrained) expects RGB input.
    """
    img = img.astype(np.float32) / 4095.0  # 12-bit max = 4095
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img_rgb = np.repeat(img, 3, axis=2)
    return img_rgb

# === Augmentation (applied only during training) ===
augmentation_gen = ImageDataGenerator(
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.1,
    fill_mode='reflect'
)

# === Custom rotation with noise-fill ===
def random_rotate_with_noise_fill(img, angle_range=20):
    """
    Rotate with constant padding, then replace padded pixels with random noise
    sampled from the image intensity distribution to avoid artificial black borders.
    """
    h, w, c = img.shape
    angle = np.random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Identify padded pixels (all channels == 0) and replace with noise background
    mask = (rotated == 0).all(axis=-1)
    mean, std = img.mean(), img.std()
    background = np.random.normal(mean, std, img.shape).astype(img.dtype)
    rotated[mask] = background[mask]
    return rotated

# === Custom generator ===
class GrayscaleToRGBDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator that:
      1) loads grayscale images from disk
      2) normalizes intensities and replicates to RGB
      3) applies augmentation only when augment=True
    """
    def __init__(self, image_filenames, labels, batch_size, img_size, shuffle=True, augment=False):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.image_filenames[i] for i in batch_indexes]
        batch_y = [self.labels[i] for i in batch_indexes]

        images = []
        for file in batch_x:
            # Load as grayscale and resize
            img = tf.keras.preprocessing.image.load_img(
                file, color_mode='grayscale', target_size=self.img_size
            )
            img = tf.keras.preprocessing.image.img_to_array(img)

            # Normalize and replicate channels
            img_rgb = preprocess_grayscale_to_rgb(img)

            # Apply augmentation only on training generator
            if self.augment:
                if np.random.rand() < 0.5:
                    img_rgb = random_rotate_with_noise_fill(img_rgb)
                img_rgb = augmentation_gen.random_transform(img_rgb)

            images.append(img_rgb)

        return np.array(images), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# === Load and split dataset (no oversampling; class weighting used instead) ===
image_paths, labels = load_dataset(DATA_DIR, CLASS_NAMES)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.1, stratify=labels, random_state=8
)

train_generator = GrayscaleToRGBDataGenerator(
    train_paths, train_labels, BATCH_SIZE, IMG_SIZE, augment=True
)
val_generator = GrayscaleToRGBDataGenerator(
    val_paths, val_labels, BATCH_SIZE, IMG_SIZE, shuffle=False, augment=False
)

# === Build model ===
# Transfer learning backbone (ImageNet-pretrained); initial phase trains only the new head
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Metrics ===
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc")
]

# === Callbacks ===
# Reduce learning rate when validation loss plateaus; stop early and restore best weights
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Initial training (frozen backbone) ===
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=METRICS
)

# Mild class weighting to bias slightly toward mature class
class_weight = {0: 1, 1: 1.1}

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[reduce_lr, early_stop]
)

# === Fine-tuning (last 5 layers trainable) ===
# Unfreeze backbone, then refreeze all but the last 5 layers and train at lower LR
base_model.trainable = True
for layer in base_model.layers[:-5]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=METRICS
)

fine_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
fine_early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=len(history.history['loss']),
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[fine_reduce_lr, fine_early_stop]
)

#%%