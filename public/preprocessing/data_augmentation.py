from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(train_directory, batch_size=32):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        train_directory,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator

# Example usage
train_gen = augment_data('data/train')
