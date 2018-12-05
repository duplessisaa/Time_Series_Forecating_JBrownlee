import tensorflow as tf 
import numpy as np 

def Split_sequences(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):
        end_indx = i + n_steps

        if end_indx > len(sequence)-1:
            break
        else:
            X.append(sequence[i:end_indx, :-1])
            y.append(sequence[end_indx-1, -1])

    return np.array(X), np.array(y)

# define the sequence
# define input sequence
in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# but we want the input seq. dims. to be rows = inexamples, columns = features
in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)

#now horizontally stack the columns
dataset = np.hstack((in_seq1,in_seq2,out_seq))

# How many time steps back do we want to rely on for the current output value(s)
n_steps = 3

X, y = Split_sequences(dataset, n_steps=3)

print("X inputs: ", X)
print("y inputs: ", y)

# Now separate the data set further
X1 = X[:, :, 0]
X2 = X[:, :, 1]

# define 1st input model
visible1 = tf.keras.layers.Input(shape=(n_steps,))
dense1 = tf.keras.layers.Dense(100, activation='relu')(visible1)

# define 2nd input model
visible2 = tf.keras.layers.Input(shape=(n_steps,))
dense2 = tf.keras.layers.Dense(100, activation='relu')(visible2)

# merge input cells
merge = tf.keras.layers.concatenate([dense1,dense2])
output = tf.keras.layers.Dense(1)(merge)

model = tf.keras.models.Model(inputs=[visible1,visible2],outputs=output)
model.compile(optimizer='adam', loss='mse')

# fit the model
model.fit(x=[X1,X2], y=y, epochs=2000, verbose=0)

x_input = np.array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps))
x2 = x_input[:, 1].reshape((1, n_steps))
y_hat = model.predict([x1, x2], verbose=0)

print('----------------')
print('prediction is : ', y_hat)
print('----------------')





