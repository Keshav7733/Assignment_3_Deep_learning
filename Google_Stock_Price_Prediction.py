#!/usr/bin/env python
# coding: utf-8

# # **Google Stock Price Prediction**

# Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Import Dataset

# In[64]:


train_data = pd.read_csv('Google_Stock_Price_Train.csv')
test_data = pd.read_csv('Google_Stock_Price_Test.csv')


# In[65]:


# Display the first few rows of the dataset
print("Training Data:")
print(train_data.head())
print("\nTesting Data:")
print(test_data.head())


# Data Preprocessing

# In[66]:


# Select relevant columns
train_data = train_data[['Open', 'High', 'Low', 'Close', 'Volume']]
test_data = test_data[['Open', 'High', 'Low', 'Close', 'Volume']]


# In[67]:


# Replace commas and convert to numeric
for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
    train_data[column] = train_data[column].astype(str).str.replace(',', '').astype(float)
    test_data[column] = test_data[column].astype(str).str.replace(',', '').astype(float)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


# In[68]:


plt.plot(train_scaled)
plt.show()


# In[69]:


plt.plot(test_scaled)
plt.show()


# In[70]:


N = 15
M = 5


# In[71]:


# Sequence creation function
def create_sequences(data, seq_length, predict_length):
    X, y = [], []
    for i in range(len(data) - seq_length - predict_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + predict_length, 3])  # 'Close'
    return np.array(X), np.array(y)

# Ensure sufficient data
min_required_rows = N + M - 1

if len(test_scaled) < min_required_rows:
    print(f"Warning: Dataset too small for the given sequence length (N={N}) and prediction length (M={M}).")
    N = max(1, len(test_scaled) - M)  # Dynamically adjust N
    print(f"Adjusted sequence length to N={N}. Check results carefully.")

# Generate sequences
X_train, y_train = create_sequences(train_scaled, N, M)
X_test, y_test = create_sequences(test_scaled, N, M)

if len(X_test) == 0:
    raise ValueError("Not enough test data to create sequences. Reduce 'N' or increase dataset size.")


# Build the LSTM Model

# In[72]:


# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(M)  # Output matches prediction length
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()


# Train the Model

# In[73]:


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)


# Plot Training History (Loss Curves)

# In[74]:


# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")


# In[75]:


# Plot training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluate on Test Data

# In[76]:


print("X_test shape:", X_test.shape)


# In[77]:


print("Test dataset shape:", test_scaled.shape)


# In[78]:


test_predicted = model.predict(X_test)
test_predicted[:5]


# In[79]:


test_inverse_predicted = scaler.inverse_transform(test_predicted) # Inversing scaling on predicted data
test_inverse_predicted[:5]


# In[80]:


# Prepare test data
real_stock_price = test_data.iloc[:,1:2].values
dataset = pd.concat((train_data[['Open', 'High', 'Low', 'Close', 'Volume']], test_data[['Open', 'High', 'Low', 'Close', 'Volume']]), axis=0)
inputs = dataset[len(dataset) - len(test_data) - 120:].values
inputs = scaler.transform(inputs)

x_test = []
for i in range(120, 140):
    # Include all 5 features in the sequence
    x_test.append(inputs[i-120:i, :])
x_test = np.array(x_test)
# Reshape to (samples, timesteps, features) - features should be 5
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], inputs.shape[1]))

# Make predictions
predicted_stock_price = model.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# Performance Evaluation

# In[81]:


# Print out results (for analysis)
print(f"Predicted stock prices for the next {len(predicted_stock_price)} days:")
print(predicted_stock_price)


# In[82]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Extract only the 'Close' price from predicted_stock_price
predicted_close_price = predicted_stock_price[:, 3]

# Reshape real_stock_price to a 1D array
real_stock_price = real_stock_price.ravel()

# Calculate evaluation metrics using the 'Close' price
mse = mean_squared_error(real_stock_price, predicted_close_price)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_stock_price, predicted_close_price)
r2 = r2_score(real_stock_price, predicted_close_price)

# Display the results
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

