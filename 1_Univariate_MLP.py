import tensorflow as tf
# from keras.models import Sequential
import numpy as np 
# from tensorflow.keras.layers import Dense

def Split_sequences(sequence, n_steps):
    X, y = list(), list()

    for i in range(len(sequence)):
        end_indx = i + n_steps
        if end_indx > len(sequence)-1: #-1 because you always want to reserve the very last digit as an y_output val
            break
        else:
            X.append(sequence[i:end_indx])
            y.append(sequence[end_indx])
    
    return np.array(X), np.array(y)

# main()
#define input sequence

raw_seq = [10,20,30,40,50,60,70,80,90]

n_steps = 3

X, y = Split_sequences(raw_seq, n_steps)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100,activation='relu',input_dim=n_steps))
model.add(tf.keras.layers.Dense(1))

#define the loss optimization routine and loss functions
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=2000, verbose=0)

# Let's predict
x_input = np.array([70,80,90])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0)
print(yhat)
