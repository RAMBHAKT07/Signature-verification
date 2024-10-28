import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
saved_model_path = '/content/Models/Models-V3/MobileNet-FineTuned-Phase-2.h5'
loaded_model = tf.keras.models.load_model(saved_model_path)

# Data generator for validation set
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 10

val_generator = val_datagen.flow_from_directory(
    '/content/split_dataset/val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Keep the order of classes to match predictions
)

# Evaluate the model on the validation set
val_loss, val_accuracy = loaded_model.evaluate(val_generator, steps=len(val_generator))

# Get the class indices
class_indices = val_generator.class_indices

# Get the predicted probabilities for each class
val_predictions = loaded_model.predict(val_generator, steps=len(val_generator))

# Threshold for validation accuracy
threshold_accuracy = 0.95

# Print the names of classes with validation accuracy lower than 95%
print("Classes with Validation Accuracy < 95%:")
for class_name, class_index in class_indices.items():
    class_accuracy = sum((val_predictions.argmax(axis=1) == class_index) & (val_generator.labels == class_index)) / sum(val_generator.labels == class_index)
    if class_accuracy < threshold_accuracy:
        print(f"{class_name}: {class_accuracy * 100:.2f}%")