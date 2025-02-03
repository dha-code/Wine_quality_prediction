import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils.model_metrics import compute_metrics, plot_loss

def read_and_split_data(filename):
  # Function to read the data and split into training and test set
  wines = pd.read_csv(filename)
  X = wines.drop(['quality'],axis=1).values
  y = wines['quality'].values
  return (train_test_split(X, y, test_size=0.2, random_state=9))

def create_model(normalizer):
  # Function to create a neural network model
  model = Sequential([
      normalizer,
      Dense(128, input_shape=(9,), activation='relu'),  
      Dropout(0.25),
      Dense(64, activation='relu'), 
      Dropout(0.25),
      Dense(32, activation='relu'), 
      Dropout(0.25),
      Dense(1)  
  ])
  # Compile the model
  optimizer = Adam(learning_rate=0.01)
  model.compile(optimizer=optimizer,
              loss='mean_squared_error', 
              metrics=['mae'])
  return model

def run_nn(filename):
  # Function to run a neural network model on the wine dataset
  X_train, X_test, y_train, y_test = read_and_split_data(filename)
  normalizer = Normalization(axis=-1)
  normalizer.adapt(np.array(X_train))
  model = create_model(normalizer)
  early_stopping = EarlyStopping(patience=20, monitor="val_loss", restore_best_weights=True)
  history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_split=0.2, 
                    callbacks=[early_stopping],
                    verbose=1)

  y_pred = model.predict(X_test)
  compute_metrics(y_test, y_pred)
  plot_loss(history.history['loss'], history.history['val_loss'])

if __name__ == "__main__":
  run_nn("./data/WinesCleaned.csv")