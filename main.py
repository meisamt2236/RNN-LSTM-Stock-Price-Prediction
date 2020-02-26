import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class StockPrediction:
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        mypath = os.getcwd()
        training_data_path = mypath + '/GOOG_train.csv'
        testing_data_path = mypath + '/GOOG_test.csv'
        self.training_data = pd.read_csv(training_data_path)
        self.testing_data = pd.read_csv(testing_data_path)
        print('\nData has been successfully initialized!\n')
    
    def show_data(self):
        print(self.training_data.tail())

    def preprocessing(self):
        training_set = self.training_data.iloc[:, 1:2].values # open column selected
        print(f'\nTraining data from {self.training_data.shape} converted to {training_set.shape}.\n')
        # Feature scaling(Normalization):
        self.sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = self.sc.fit_transform(training_set)
        print('Training set has been scaled from 0 to 1.\n')
        for i in range(60, training_set.shape[0]):
            self.x_train.append(training_set_scaled[i-60:i, 0])
            self.y_train.append(training_set_scaled[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        print('Created a data structure with 60 timesteps and one output.\n')
        # RNN takes 3 dimensions so reshaping:
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 60, 1)

    def model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True, input_shape=(60, 1))) # first LSTM layer
        self.model.add(tf.keras.layers.Dropout(0.2)) # dropout layer (regularization technique)
        self.model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True)) # second LSTM layer
        self.model.add(tf.keras.layers.Dropout(0.2)) # dropout layer (regularization technique)
        self.model.add(tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True)) # third LSTM layer
        self.model.add(tf.keras.layers.Dropout(0.2)) # dropout layer (regularization technique)
        self.model.add(tf.keras.layers.LSTM(units=120, activation='relu')) # last LSTM layer
        self.model.add(tf.keras.layers.Dropout(0.2)) # dropout layer (regularization technique)
        self.model.add(tf.keras.layers.Dense(units=1))
        print(f'Model created: \n{self.model.summary()}\n')
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print('Model compiled!\n')
    
    def training_model(self):
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=100)
        print('Model trained!')
    
    def save_model(self):
        self.model.save("model.h5")
        print('Model has been saved in the directory!')

    def load_model(self):
        self.model = tf.keras.models.load_model('model.h5')
        print('Model has benn successfully loaded!')
    
    def testing_model(self):
        self.testing_set = self.testing_data.iloc[:, 1:2].values # open column selected
        print(f'\nTesting data from {self.testing_data.shape} converted to {self.testing_set.shape}.\n')
        dataset_total = pd.concat((self.training_data['Open'], self.testing_data['Open']), axis=0) # concatinating
        inputs = dataset_total[len(dataset_total) - len(self.testing_data) - 60:].values
        inputs = inputs.reshape(-1, 1) # converted to numpy array
        # Feature scaling(Normalization):
        inputs = self.sc.transform(inputs)
        for i in range(60, self.testing_data.shape[0]):
            self.x_test.append(inputs[i-60:i, 0])
        self.x_test = np.array(self.x_test)
        # RNN takes 3 dimensions so reshaping:
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))
        self.predicted_stock_price = self.model.predict(self.x_test)
        self.predicted_stock_price = self.sc.inverse_transform(self.predicted_stock_price)
        print('Prediction proceed!')
        for i in range(1, 11):
            print(f'Day {i} actual price: {self.testing_set[i]} | predicted price: {self.predicted_stock_price[i]}\n')

    def visualization(self):
        plt.plot(self.testing_set, color='green', label='Actual Google Stock Price')
        plt.plot(self.predicted_stock_price, color='blue', label='Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    prediction = StockPrediction()
    prediction.show_data()
    prediction.preprocessing()
    #prediction.model()
    #prediction.training_model()
    #prediction.save_model()
    prediction.load_model()
    prediction.testing_model()
    prediction.visualization()

if __name__ == '__main__':
    main()