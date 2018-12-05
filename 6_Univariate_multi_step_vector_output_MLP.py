'''
We will look at a vector output that represents multiple time steps of one variable, for example:
In = [10 20 30] out = [40 50]
In = [20 30 40] out = [50 60]
In = [30 40 50] out = [60 70]
'''

import tensorflow as tf 
import numpy as np 

def Split_sequence(seq, n_steps_in, n_steps_out):
    X, y = list(), list()

    for i in range(len(seq)):
        end_indx = i + n_steps_in

        if end_indx > (len(seq) - n_steps_out):
            break
        else:
            X.append(seq[i:end_indx])
            y.append(seq[end_indx:end_indx + n_steps_out])
    return np.array(X), np.array(y)

#define raw sequence
raw_seq = [10, 20 ,30 ,40 ,50 ,60 ,70, 80, 90]

# number of time steps
n_steps_in = 3
n_steps_out = 2

#split data into samples
X, y = Split_sequence(seq=raw_seq, n_steps_in=n_steps_in, n_steps_out=n_steps_out)


# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_dim=n_steps_in))
model.add(tf.keras.layers.Dense(n_steps_out))

#define the optimizer and the loss func.
model.compile(optimizer='adam', loss='mse')


# train the model
model.fit(X, y, epochs=2000, verbose=0)

# Demonstrate prediction
x_in = np.array([70, 80, 90])
x_in = x_in.reshape((1, n_steps_in)) 

y_hat = model.predict(x_in, verbose=0)

print("Answer : 100, 110")
print("Prediction : ", y_hat)
