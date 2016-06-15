"""Based on examples from tensorflow source"""

from time import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

class RNNModel():
    def __init__(self, config):
        sent_len = config.sent_len
        batch_size = config.batch_size
        vocab_size = config.vocab_size
        embed_size = config.embed_size
        num_layers = config.num_layers
        state_size = config.state_size
        keep_prob = config.keep_prob

        self.input_data = tf.placeholder(tf.int32, [batch_size, sent_len])
        self.lengths = tf.placeholder(tf.int64, [batch_size])
        self.targets = tf.placeholder(tf.float32, [batch_size, 1])

        # Get embedding layer which requires CPU
        with tf.device("/cpu:0"):
            embeding = tf.get_variable("embeding", [vocab_size, embed_size])
            inputs = tf.nn.embedding_lookup(embeding, self.input_data)

        #LSTM 1 -> Encode the characters of every tok into a fixed dense representation
        with tf.variable_scope("rnn1", reuse=None):
            cell = rnn_cell.LSTMCell(state_size, input_size=embed_size)
            back_cell = rnn_cell.LSTMCell(state_size, input_size=embed_size)
            cell = rnn_cell.DropoutWrapper(
              cell, input_keep_prob=keep_prob,
                         output_keep_prob=keep_prob)
            back_cell = rnn_cell.DropoutWrapper(
              back_cell, input_keep_prob=keep_prob,
                              output_keep_prob=keep_prob) 
            cell = rnn_cell.MultiRNNCell([cell] * num_layers)
            backcell = rnn_cell.MultiRNNCell([back_cell] * num_layers)
            
            rnn_splits = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sent_len, inputs)]

            # Run the bidirectional rnn
            outputs, last_fw_state, last_bw_state = rnn.bidirectional_rnn(
                                                        cell, backcell, rnn_splits,
                                                        sequence_length=self.lengths,
                                                        dtype=tf.float32)
        
        sent_out = tf.concat(1, [last_fw_state, last_bw_state])
        #sent_out = outputs[-1]
        output_size = state_size*4

        with tf.variable_scope("linear", reuse=None):
            w = tf.get_variable("w", [output_size, 1])
            b = tf.get_variable("b", [1])
            raw_logits = tf.matmul(sent_out, w) + b 
        self.probabilities = tf.sigmoid(raw_logits)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(raw_logits, self.targets))

        #Calculate gradients and propagate
        #Aggregation method 2 is really important for rnn per the tensorflow issues list
        tvars = tf.trainable_variables()
        self.lr = tf.Variable(0.0, trainable=False) #Assign to overwrite
        optimizer = tf.train.GradientDescentOptimizer(self.lr) 
        grads, _vars = zip(*optimizer.compute_gradients(self.cost, tvars, aggregation_method=2))
        grads, self.grad_norm = tf.clip_by_global_norm(grads,
                                      config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, _vars))

    def run_epoch(self, session, reader, training):
        """Run one complete pass over the training data. 

        Args:
            session: tf session
            m: model object
            reader: open reader stream (iterator)

        Returns:
            The total cost for the epoch, the median and max costs for the
            batches in the epoch.

        """

        t0 = time()
        total_cost = 0.0
        costs = []
        accuracy = []
        for step, (x,y,lengths) in enumerate(reader):
            num_data_points = len(x)
            feed_dict = {self.input_data:x, self.targets:y,
                         self.lengths:lengths}
            if training:
                fetches =  [self.cost, self.grad_norm, self.train_op]
                cost, grad_norm, _  = session.run(fetches, feed_dict)
                total_cost += cost
                costs.append(cost)
                print("%.3f cost: %.3f grad norm: %.3f speed: %.0f wps" %
                    (step, cost, grad_norm,
                     (num_data_points / float(time() - t0))))
                t0 = time()

            else:
                print("Test step: ",step)
                fetches =  self.probabilities
                proba = session.run(fetches, feed_dict) 
                choice = np.where(proba > 0.5, 1, 0)
                accuracy.append(np.mean(choice == y))



        if training:
            return total_cost, np.median(costs), np.max(costs)
        return np.mean(accuracy)
   



class CNNModel():
    pass


class RNNConvModel():
    pass

class ConvRNNModel():
    pass






 





