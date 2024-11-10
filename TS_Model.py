# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Assuming you already have loaded data1 and data2
# Example data loading (you will replace this with actual file loading)
data1 = pd.read_csv('preprocessed_space_data.csv')  # Replace with actual file path
data2 = pd.read_csv('space_decay.csv')  # Replace with actual file path

# Step 1: General EDA for data1 and data2

# Display basic information about data1 and data2
print("Data1 Info:")
print(data1.info())
print("\nData2 Info:")
print(data2.info())

# Display summary statistics for both datasets
print("\nData1 Summary Statistics:")
print(data1.describe())
print("\nData2 Summary Statistics:")
print(data2.describe())

# Step 2: Missing values analysis
print("\nData1 Missing Values:")
print(data1.isnull().sum())
print("\nData2 Missing Values:")
print(data2.isnull().sum())

# Step 3: Visualizing distributions for numerical columns in both datasets
# For simplicity, only select a few numerical columns here

# Select numerical columns for visualization
numerical_cols_data1 = data1.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_data2 = data2.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms for numerical columns in data1
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data1[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data1[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Plot histograms for numerical columns in data2
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_data2[:9]):  # Limiting to the first 9 columns
    plt.subplot(3, 3, i + 1)
    sns.histplot(data2[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Step 4: Correlation analysis
# Correlation matrix for data1
correlation_matrix_data1 = data1[numerical_cols_data1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data1, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Data1')
plt.show()

# Correlation matrix for data2
correlation_matrix_data2 = data2[numerical_cols_data2].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_data2, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Data2')
plt.show()

# Step 5: Exploring categorical data
# Show value counts for categorical columns in data1
categorical_cols_data1 = data1.select_dtypes(include=['object']).columns
for col in categorical_cols_data1:
    print(f"\nValue counts for {col} in Data1:")
    print(data1[col].value_counts())

# Show value counts for categorical columns in data2
categorical_cols_data2 = data2.select_dtypes(include=['object']).columns
for col in categorical_cols_data2:
    print(f"\nValue counts for {col} in Data2:")
    print(data2[col].value_counts())

# Step 6: Visualizing sample images (if applicable)

# If you have images associated with the data, you can plot a sample of images as follows
# (Here assuming you have a dataset that has image data)

# Function to plot a batch of images
def plot_images(data_batch):
    images, labels = next(data_batch)
    plt.figure(figsize=(10, 10))
    
    # Loop through only the first 9 images
    for i in range(9):  # Limiting to 9 images for a 3x3 grid
        plt.subplot(3, 3, i + 1)  # Corrected: using i+1 to avoid starting at 0
        plt.imshow(images[i])
        plt.title(f"Class: {labels[i].argmax()}")
        plt.axis('off')
    
    plt.show()


# Step 7: Data Preparation for LSTM Model

# Select relevant columns for time series forecasting
# (e.g., columns that might be indicative of collision patterns, such as MEAN_MOTION, ECCENTRICITY, INCLINATION)
ts_features = ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY']

# Filter out data for these selected features
ts_data = data1[ts_features].copy()

# Handle missing values by forward filling (or use another imputation method)
ts_data.fillna(method='ffill', inplace=True)


# Step 8: Scaling the Data
# Scale the data to a range between 0 and 1 for better LSTM performance
scaler = MinMaxScaler(feature_range=(0, 1))
ts_data_scaled = scaler.fit_transform(ts_data)

# Step 9: Create Sequences for Time Series Data

# Define a function to create sequences of features for LSTM input
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :])
        y.append(data[i + sequence_length, 0])  # Predicting MEAN_MOTION as target for simplicity; adjust as needed
    return np.array(X), np.array(y)

# Define sequence length (e.g., using last 30 time steps to predict the next)
sequence_length = 30
X, y = create_sequences(ts_data_scaled, sequence_length)


# Step 10: Splitting the Data into Training and Test Sets

# Define train/test split (e.g., 80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Step 11: Build the LSTM Model

# Define the model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 12: Train the Model

# Early stopping callback to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Step 13: Evaluate the Model

# Model evaluation on test data
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Step 14: Make Predictions and Inverse Scale

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse scale the predictions and the true values to get them back to original scale
y_pred_inverse = scaler.inverse_transform(np.concatenate((y_pred, X_test[:, -1, 1:]), axis=1))[:, 0]
y_test_inverse = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

# Step 15: Plotting Actual vs Predicted Values

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, color='blue', label='Actual Values')
plt.plot(y_pred_inverse, color='red', label='Predicted Values')
plt.title('LSTM Model - Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('MEAN_MOTION (or selected target)')
plt.legend()
plt.show()

# Saving it as .h5 file
model.save('TS_Model'.h5')