from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(y_test, y_pred):
    # Function to compute metrics based on predicted values and actual values
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Print results
    print("\n**MODEL EVALUATION METRICS**")
    print("-----------------------------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

def plot_loss(loss,val_loss):
    # Function to plot the loss of train vs test 
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig("./figures/Train_vs_test.png")
    print("Image saved to ./figures/Train_vs_test.png")