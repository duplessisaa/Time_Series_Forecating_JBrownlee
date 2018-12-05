import tensorflow as tf 
import numpy as np 

def Split_sequences(seqs, n_steps):

    X, y = list(), list()

    for i in range(len(seqs)):
        end_indx = i + n_steps # here we find the index of the last value to serve as input

        if end_indx > len(seqs)-1: # check for the end of data set
            break
        else:                      # create the input and output data set
            X.append(seqs[i:end_indx, : ])  # input data
            y.append(seqs[end_indx, :])     # output values

    return np.array(X), np.array(y)

# define input sequence
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = np.hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps
n_steps = 3

# convert into input/output
X, y = Split_sequences(dataset, n_steps)
print(X.shape, y.shape)

# summarize the data
for i in range(len(X)):
    print(X[i], y[i])

# Prepare the data for MLP
# Flatten the data

n_input = X.shape[1] * X.shape[2]       # the nr of columns*each column entry (all act as features)
X = X.reshape((X.shape[0], n_input))

n_output = y.shape[1]

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_dim=n_input))
model.add(tf.keras.layers.Dense(n_output))

#Define optimizer and loss function
model.compile(optimizer='adam', loss='mse')

# train the model
model.fit(x=X, y=y, epochs=2000, verbose=0)

# Make some predictions
x_in = np.array([[70,75,145], [80,85,165], [90,95,185]])
x_in = x_in.reshape((1, n_input))
y_hat = model.predict(x_in, verbose=0)

print('Prediction should be : [100, 105, 205]')
print('Prediction is : ', y_hat)