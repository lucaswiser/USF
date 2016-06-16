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
            cell = rnn_cell.LSTMCell(state_size, input_size=embed_size, initializer=tf.contrib.layers.xavier_initializer())
            back_cell = rnn_cell.LSTMCell(state_size, input_size=embed_size, initializer=tf.contrib.layers.xavier_initializer())
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
        #sent_out = tf.add_n(outputs)
        output_size = state_size*4

        with tf.variable_scope("linear", reuse=None):
            w = tf.get_variable("w", [output_size, 1])
            b = tf.get_variable("b", [1], initializer=tf.constant_initializer(0.0))
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
                print("%.3f cost: %.3f grad norm: %.3f speed: %.0f pages/sec" %
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
    def __init__(self, config):
        sent_len = config.sent_len
        batch_size = config.batch_size
        vocab_size = config.vocab_size
        embed_size = config.embed_size
        filter_sizes = config.filter_sizes
        num_filters = config.num_filters
        if len(num_filters) == 1:
            num_filters = num_filters*len(filter_sizes)
        output_size = sum(num_filters)
        keep_prob = config.keep_prob

        self.input_data = tf.placeholder(tf.int32, [batch_size, sent_len])
        self.targets = tf.placeholder(tf.float32, [batch_size, 1])

        # Get embedding layer which requires CPU
        with tf.device("/cpu:0"):
            embeding = tf.get_variable("embeding", [vocab_size, embed_size])
            inputs = tf.nn.embedding_lookup(embeding, self.input_data)
            inputs_expanded = tf.expand_dims(inputs, -1)


        pooled_outputs = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" %(filter_size)):
                filter_shape = [filter_size, embed_size, 1, num_filters[i]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, name="W"))
                b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name="b")
                conv = tf.nn.conv2d(
                    inputs_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sent_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, output_size])
        conv_output = tf.nn.dropout(h_pool_flat, config.keep_prob)


        with tf.variable_scope("linear", reuse=None):
            w = tf.get_variable("w", [output_size, 1])
            b = tf.get_variable("b", [1])
            raw_logits = tf.matmul(conv_output, w) + b 
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
            feed_dict = {self.input_data:x, self.targets:y}
            if training:
                fetches =  [self.cost, self.grad_norm, self.train_op]
                cost, grad_norm, _  = session.run(fetches, feed_dict)
                total_cost += cost
                costs.append(cost)
                print("%.3f cost: %.3f grad norm: %.3f speed: %.0f pages/sec" %
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

class RNNRNNModel():
    def __init__(self, config):
        sent_len = self.sent_len = config.sent_len
        word_len = config.word_len
        batch_size = config.batch_size
        vocab_size = config.vocab_size
        embed_size = config.embed_size
        keep_prob1 = config.keep_prob1
        keep_prob2 = config.keep_prob2
        num_layers1 = config.num_layers1
        num_layers2 = config.num_layers2
        state_size1 = config.state_size1
        state_size2 = config.state_size2

        self.input_data = tf.placeholder(tf.int32, [batch_size*sent_len, word_len])
        self.lengths = tf.placeholder(tf.int64,[batch_size])
        self.wordlengths = tf.placeholder(tf.int64, [batch_size*sent_len])
        self.targets = tf.placeholder(tf.float32, [batch_size, 1])

        # Get embedding layer which requires CPU
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embed_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        #LSTM 1 -> Encode the characters of every tok into a fixed dense representation
        with tf.variable_scope("rnn1", reuse=None):
            lstm_cell_1 = rnn_cell.LSTMCell(state_size1, input_size=embed_size)
            lstm_back_cell_1 = rnn_cell.LSTMCell(state_size1, input_size=embed_size)
            if keep_prob1 < 1:
                #Only on the inputs for rnn1. That way we don't dropout twice 
                lstm_cell_1 = rnn_cell.DropoutWrapper(
                  lstm_cell_1, input_keep_prob=keep_prob1)
                lstm_back_cell_1 = rnn_cell.DropoutWrapper(
                  lstm_back_cell_1, input_keep_prob=keep_prob1)

            cell_1 = rnn_cell.MultiRNNCell([lstm_cell_1] * num_layers1)
            backcell_1 = rnn_cell.MultiRNNCell([lstm_back_cell_1] * num_layers1)
            
            rnn_splits = [tf.squeeze(input_, [1]) for input_ in tf.split(1, word_len, inputs)]


            # Run the bidirectional rnn
            outputs1, last_fw_state1, last_bw_state1 = rnn.bidirectional_rnn(
                                                        cell_1, backcell_1, rnn_splits,
                                                        sequence_length=self.wordlengths,
                                                        dtype=tf.float32)

        #tok_embeds = outputs1[-1]
        tok_embeds = tf.concat(1, [last_fw_state1, last_bw_state1])
        
        with tf.variable_scope("rnn2", reuse=None):
            lstm_cell_2 = rnn_cell.LSTMCell(state_size2, input_size=state_size1*4)
            lstm_back_cell_2 = rnn_cell.LSTMCell(state_size2, input_size=state_size1*4)
            # Add dropout. NOTE: this adds to the input and output layers. Remember that the input layer
            # is the output from the conv net, so this also adds dropout to the output of the conv net
            if keep_prob2 < 1:
                lstm_cell_2 = rnn_cell.DropoutWrapper(
                  lstm_cell_2, input_keep_prob=keep_prob2,
                             output_keep_prob=keep_prob2)
                lstm_back_cell_2 = rnn_cell.DropoutWrapper(
                  lstm_back_cell_2, input_keep_prob=keep_prob2,
                                  output_keep_prob=keep_prob2) 

            cell_2 = rnn_cell.MultiRNNCell([lstm_cell_2] * num_layers2)
            backcell_2 = rnn_cell.MultiRNNCell([lstm_back_cell_2] * num_layers2)

            # The rnn synthesis of the tokens is size [batch_size*sent_len, state_size*2]
            # we want it to be a list of sent_len of [batch_size, state_size*2]
            # We partition as [0,1,2,...n,0,1,2,...n...]
            rnn_inputs2 = tf.dynamic_partition(tok_embeds, list(range(sent_len))*batch_size, sent_len)
            

            #Sent level rnn
            outputs2, last_fw_state2, last_bw_state2 = rnn.bidirectional_rnn(cell_2, backcell_2, rnn_inputs2,
                                                                        sequence_length=self.lengths,
                                                                        dtype=tf.float32)
            #sent_embed = tf.reshape(tf.concat(1, [last_fw_state2, last_bw_state2]), [batch_size, state_size2*4])
            sent_embed = tf.concat(1, [last_fw_state2, last_bw_state2])

        with tf.variable_scope("linear", reuse=None):
            w = tf.get_variable("w", [state_size2*4, 1])
            b = tf.get_variable("b", [1])
            raw_logits = tf.matmul(sent_embed, w) + b 
        self.probabilities = tf.sigmoid(raw_logits)
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(raw_logits, self.targets))

        #Calculate gradients and propagate
        #Aggregation method 2 is really important for rnn per the tensorflow issues list
        tvars = tf.trainable_variables()
        self.lr = tf.Variable(0.0, trainable=False) #Assign to overwrite
        optimizer = tf.train.AdamOptimizer()
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
        for step, (x,y,lengths,wordlengths) in enumerate(reader):
            num_data_points = len(x)/self.sent_len
            feed_dict = {self.input_data:x, self.targets:y, self.wordlengths:wordlengths,
                         self.lengths:lengths}
            if training:
                fetches =  [self.cost, self.grad_norm, self.train_op]
                cost, grad_norm, _  = session.run(fetches, feed_dict)
                total_cost += cost
                costs.append(cost)
                print("%.3f cost: %.3f grad norm: %.3f speed: %.0f pages/sec" %
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

class RNNConvModel():
    pass







 





