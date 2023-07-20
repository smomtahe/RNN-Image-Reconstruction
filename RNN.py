import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import SimpleRNN, Dense, Reshape
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Load data from CSV file
xtrain_url = "https://raw.githubusercontent.com/smomtahe/ensemble-learning-absorption/main/phantom.csv"
data = pd.read_csv(xtrain_url)

# Select features
xtrain_features = ['reflectance1', 'reflectance2']

# Group the data into 8 equal parts and calculate the average
grouped_data = data.groupby(np.arange(len(data)) // 16, as_index=False).mean()

X_train = grouped_data[xtrain_features].values

# Load and preprocess the y train image
image_path = "G:\Dr Golnaraghi\Directed Studies Course\Research Reports\7-Maryam-RCB-ua paper IEEE Journal Nemo/image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    scaler = MinMaxScaler()
    image_scaled = scaler.fit_transform(image.reshape(-1, 1))
else:
    print("Failed to load the image.")

# Normalize the input features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Reshape input data for RNN
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)

# Create the RNN model
model = Sequential()
model.add(SimpleRNN(64, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64))
model.add(Dense(8*8, activation='sigmoid'))  # Output layer with flattened shape

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
if image is not None:
    model.fit(X_train_reshaped, image_scaled, epochs=10, batch_size=32)
else:
    print("Skipping model training due to image loading failure.")

# Load and preprocess the test set
xtest_url = "https://raw.githubusercontent.com/maryammomtahen/Testset/main/P14_S4_U_post.csv"
datatest = pd.read_csv(xtest_url)

X_test = datatest[xtrain_features].values

# Normalize the test set features
X_test_scaled = scaler.transform(X_test)

# Reshape the test set for prediction
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Predict the output for the test set
y_pred = model.predict(X_test_reshaped)

# Reshape the predicted output into images
predicted_images = y_pred.reshape(y_pred.shape[0], 8, 8)

# Display the first predicted image
plt.imshow(predicted_images[0], cmap='gray')
plt.axis('off')
plt.show()
