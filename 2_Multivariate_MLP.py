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

#define the sequence
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

X, y = Split_sequences(dataset, n_steps=3)

print("X inputs: ", X)
print("y inputs: ", y)

# Construct the MLP model
# First flatten the input data
n_input = X.shape[1]*X.shape[2]
X = X.reshape((X.shape[0], n_input))


#Define the multivariate model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(50,input_dim=n_input,activation='relu'))
model.add(tf.keras.layers.Dense(50,activation='relu'))
model.add(tf.keras.layers.Dense(1))

#Define the optimizer and the loss finc.
model.compile(optimizer='adam',loss='mse')

# Train model
model.fit(X, y, epochs=2000, verbose=0)

#Demonstrate the model's ability
x_input = np.array([[80,85],[90, 95], [100, 105]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)

X = X.reshape((X.shape[0], n_input))

print('----------------')
print('prediction is : ', yhat)
print('----------------')





