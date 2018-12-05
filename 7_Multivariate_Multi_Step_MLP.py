"""
There are those multivariate time series forecasting problems where the output series is separate
but dependent upon the input time series, and multiple time steps are required for the output
series.
"""

import tensorflow as tf 
import numpy as np 

def Split_sequences(seq, n_steps_in, n_steps_out):
    X, y = list(), list()

    for i in range(len(seq)):
        end_idx = i + n_steps_in

        if end_idx > len(seq) - n_steps_out:
            break
        else:
            X.append(seq[i:end_idx, :-1])
            y.append(seq[end_idx -1 : end_idx -1 + n_steps_out, -1])
    return np.array(X), np.array(y)

# define input sequence
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# convert structure to rows and cols
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = np.hstack((in_seq1, in_seq2, out_seq))

# Define the nr of time-series steps
n_steps_in = 3          # How many historic data entries influence our prediction ?
n_steps_out = 2         # How far into the future will we predict ?

X, y = Split_sequences(dataset, n_steps_in, n_steps_out)

nr_inputs = X.shape[1] * X.shape[2]

X = X.reshape((X.shape[0], nr_inputs))

# Define the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(nr_inputs,)))
model.add(tf.keras.layers.Dense(n_steps_out))

# Define the optimizer and the loss function
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=2000, verbose=0)

# Predictions
x_in = np.array([[70, 75], [80, 85], [90, 95]])

x_in = x_in.reshape((1, nr_inputs))

y_hat = model.predict(x_in, verbose=0)

print("y : ", [185, 205])
print(y_hat)
