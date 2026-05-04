import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------------
# 1. Image Preprocessing
# -------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# -------------------------------------------------
# 2. Load Dataset (YOUR PATH)
# -------------------------------------------------

train_data = train_datagen.flow_from_directory(
    r'C:\Users\Swanand\OneDrive\Desktop\DL PROGRAM\dataset\train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    r'C:\Users\Swanand\OneDrive\Desktop\DL PROGRAM\dataset\validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------------------------------
# 3. Build CNN Model
# -------------------------------------------------

model = Sequential()

# Layer 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Fully Connected
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))   # Prevent overfitting

# Output Layer
model.add(Dense(train_data.num_classes, activation='softmax'))

# -------------------------------------------------
# 4. Compile Model
# -------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------
# 5. Train Model
# -------------------------------------------------

history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# -------------------------------------------------
# 6. Evaluate Model
# -------------------------------------------------

loss, accuracy = model.evaluate(test_data)
print("\nFinal Accuracy:", accuracy)

# -------------------------------------------------
# 7. Save Model (Optional but useful)
# -------------------------------------------------

model.save("plant_disease_model.h5")
