# Importing modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import setuptools.dist
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import ast
import matplotlib.patches as patches

# File paths
train_csv_path = 'debris-detection/train.csv'  # Replace with the actual path
val_csv_path = 'debris-detection/val.csv'      # Replace with the actual path
train_image_folder = 'debris-detection/train'  # Replace with the actual path to the train images folder
val_image_folder = 'debris-detection/val'      # Replace with the actual path to the val images folder

# Dataset loading 
data1 = pd.read_csv(train_csv_path)
data2 = pd.read_csv(val_csv_path)

# Step 1: General EDA for datasets
# Displaying basic information from datasets
print("Training Data Info:")
print(data1.info())
print("\nValidation Data Info:")
print(data2.info())

# Displaying statistics for both datasets
print("\nTraining Data Summary Statistics:")
print(data1.describe())
print("\nValidation Data Summary Statistics:")
print(data2.describe())

# Step 2: Analyzing the missing values
print("\nTraining Data Missing Values:")
print(data1.isnull().sum())
print("\nValidation Data Missing Values:")
print(data2.isnull().sum())

# Step 3: Visualizing distributions for numerical columns in both datasets
# Selecting the numerical columns for visualization
numerical_cols_data1 = data1.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_data2 = data2.select_dtypes(include=['float64', 'int64']).columns

# Plotting histograms for numerical columns in training data
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data1[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data1[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Plotting histograms for numerical columns in validation data
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data2[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data2[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Step 4: Feature Engineering and Correlation analysis
# Correlation matrix for training data
correlation_matrix_data1 = data1[numerical_cols_data1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data1, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Training Data')
plt.show()

# Correlation matrix for validation data
correlation_matrix_data2 = data2[numerical_cols_data2].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data2, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Validation Data')
plt.show()

# Step 5: Exploring categorical data
# Showing value counts for categorical columns in training data
categorical_cols_data1 = data1.select_dtypes(include=['object']).columns
for col in categorical_cols_data1:
    print(f"\nValue counts for {col} in Training Data:")
    print(data1[col].value_counts())

# Showing value counts for categorical columns in validation data
categorical_cols_data2 = data2.select_dtypes(include=['object']).columns
for col in categorical_cols_data2:
    print(f"\nValue counts for {col} in Validation Data:")
    print(data2[col].value_counts())

# Create a function to load the images and resize them
def load_image(image_id, base_path, target_size=(224, 224)):
    img_path = os.path.join(base_path, f"{image_id}.jpg")  # Adjust if the file format is different
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image {img_path} not found.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize the image to [0, 1]
    return img


# Set a maximum number of bounding boxes per image
max_boxes = 4

# Helper function to pad or trim bounding boxes to match max_boxes
def pad_or_trim_bboxes(bboxes, max_boxes=4):
    # If there are fewer than max_boxes, pad with [0, 0, 0, 0]
    if len(bboxes) < max_boxes:
        bboxes += [[0, 0, 0, 0]] * (max_boxes - len(bboxes))
    # If there are more than max_boxes, trim to max_boxes
    return bboxes[:max_boxes]


# Example of loading images and bounding boxes
def load_data(annotations, base_path):
    images = []
    bboxes = []
    for index, row in annotations.iterrows():
        print(f"Processing row {index + 1}/{len(annotations)}: {row['ImageID']}")
        image_id = row['ImageID']
        bbox_list = ast.literal_eval(row['bboxes'])  # Convert string to list
        
        # Apply padding or trimming to bounding boxes
        bbox_list = pad_or_trim_bboxes(bbox_list, max_boxes)
        
        # Load image and normalize
        try:
            img = load_image(image_id, base_path)
        except FileNotFoundError as e:
            print(e)
            continue  # Skip this image if not found
        images.append(img)
        
        # Flatten bounding boxes for model training
        flattened_bboxes = [coord for bbox in bbox_list for coord in bbox]
        bboxes.append(flattened_bboxes)

    return np.array(images), np.array(bboxes)

# Load the images and bounding boxes
images, bboxes = load_data(data1, train_image_folder)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, bboxes, test_size=0.2, random_state=42)

print(X_train)

# Debugging: Print shapes and sample data
print("Shape of X_train:", X_train.shape)  # Should be (num_samples, 224, 224, 3)
print("Shape of y_train:", y_train.shape)  # Should be (num_samples, max_boxes * 4)
print("Shape of X_val:", X_val.shape)      # Should be (num_samples, 224, 224, 3)
print("Shape of y_val:", y_val.shape)      # Should be (num_samples, max_boxes * 4)

# Sample data check
print("\nSample bounding box data (first entry in y_train):", y_train[0])

# Defining CNN Model
model = Sequential()
print(model.summary())

# First Convolutional Block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output to feed it into the fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output layer (4 values for bounding box coordinates)
model.add(Dense(max_boxes * 4, activation='linear'))  # Regression output for bounding boxes

# Compile the model
model.compile(optimizer=Adam(), loss='mse')  # Mean Squared Error for regression tasks

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Example: Predicting bounding boxes for validation images
predictions = model.predict(X_val)


# Visualize the predictions alongside the actual bounding boxes
def visualize_predictions(images, true_bboxes, predicted_bboxes, n=4):
    plt.figure(figsize=(15, 15))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        img = images[i]
        plt.imshow(img)
        
        # Reshape bounding boxes to (max_boxes, 4)
        true_bboxes_reshaped = true_bboxes[i].reshape((max_boxes, 4))
        predicted_bboxes_reshaped = predicted_bboxes[i].reshape((max_boxes, 4))
        
        # Draw the true bounding boxes (green)
        for bbox in true_bboxes_reshaped:
            x_min, y_min, x_max, y_max = bbox
            if [x_min, y_min, x_max, y_max] != [0, 0, 0, 0]:  # Skip padded boxes
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)
        
        # Draw the predicted bounding boxes (red)
        for bbox in predicted_bboxes_reshaped:
            x_min, y_min, x_max, y_max = bbox
            if [x_min, y_min, x_max, y_max] != [0, 0, 0, 0]:  # Skip padded boxes
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
        
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize the predictions for the first 5 images
visualize_predictions(X_val, y_val, predictions)