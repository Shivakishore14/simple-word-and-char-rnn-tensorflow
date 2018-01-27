import tensorflow as tf
import numpy as np
import random
from tensorflow.contrib import rnn


filename = __file__

logs_path = "./logs/char-rnn"
writer = tf.summary.FileWriter(logs_path)

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]  # remove start and end spaces in each line
    content = [list(i) for i in content]    # spliting into chrs
    content = np.hstack(content)            # flatteing the array to 1d
    return content


training_data = read_data(filename)
print("Loaded training data...")
vocab = set(training_data)

dictionary = {data: i for i, data in enumerate(vocab)}
reverse_dictionary = {i: data for i, data in enumerate(vocab)}

vocab_size = len(vocab)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 500
n_input = 4

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])


def RNN(x, n_hidden, vocab_size):
    # RNN output node weights and biases
    weight = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    bias = tf.Variable(tf.random_normal([vocab_size]))

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    x = tf.split(x, n_input, 1)

    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias


pred = RNN(x, n_hidden, vocab_size)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input+1)
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-(n_input + 1)):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))

        step += 1
        offset += (n_input+1)
    print("Training finished")