import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wines = pd.read_csv('./data/WineQT.csv')
wines = wines.drop(['Id'],axis=1)
print(wines.duplicated().value_counts())

X = wines.drop(['quality'],axis=1)
y = wines['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
print(X_train.shape)

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(32, activation='relu'), 
    Dense(1)  
])
# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error', 
              metrics=['mae'])

history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=256, 
                    validation_data=(X_test, y_test), 
                    verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

plot_loss(history.history['loss'], history.history['val_loss'])