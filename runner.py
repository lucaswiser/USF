import os
import tensorflow as tf
import numpy as np
import pickle
import sys
import argparse
import logging
from config import Config
from model import TokModel
from reader import TokReader
graph_path = 'graphs/'
model_dir = "models/"
with open('tok_map.pkl', 'rb') as f:
    tok_map = pickle.load(f)
tokreader = TokReader(Config.sent_len, Config.batch_size, tok_map, random=True, rounded=True, training=True)
validtokreader = TokReader(Config.sent_len, Config.batch_size, tok_map, random=True, rounded=True, training=False)

def main(graph_path, continue_training=False, start_model=None, start_ind=0):
    """Run a complete training session. Will load a saved model to continue training
    if provided. After every epoch the current model will be saved, and the tensorboard
    will graph new data.
    """  
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-Config.init_scale,
                                                     Config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = TokModel(config=Config)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(max_to_keep=Config.num_models)
        if continue_training:
            print("Continuing training from saved model ",start_model)
            saver.restore(session,start_model)
        writer = tf.train.SummaryWriter(graph_path, max_queue=1) 
        last3 = []
        learning_rate = Config.learning_rate
        session.run(tf.assign(m.lr, learning_rate))
        tol = 0.001
        for i in range(start_ind, start_ind+Config.num_epochs):
            print("EPOCH: %s"%i)
            print("learning_rate: %s"%learning_rate)
            epoch_cost, median_cost, max_cost = m.run_epoch(session, tokreader.get_sents(), True)   
            print("Total cost for EPOCH: %s"%i)
            print(epoch_cost)
            print("Median cost: %s"%median_cost)
            print("Max cost: %s"%max_cost)
            accuracy = m.run_epoch(session, validtokreader.get_sents(), False)
            print("accuracy: %s"%accuracy)
            summ1 = tf.scalar_summary("epoch_cost", tf.constant(epoch_cost))
            summ2 = tf.scalar_summary("median_cost", tf.constant(median_cost))
            summ3 = tf.scalar_summary("max_cost", tf.constant(max_cost))
            summ4 = tf.scalar_summary("learning_rate", tf.constant(learning_rate))
            summ5 = tf.scalar_summary("accuracy", tf.constant(accuracy))
            merge = tf.merge_summary([summ1, summ2, summ3, summ4, summ5])
            writer.add_summary(merge.eval(), i)
            writer.flush()
            if i % 5 == 0:
                saver.save(session, model_dir + 'saved-lstm-model', global_step=i)
            if len(last3) == 3:
                h = max(last3)
                if last3[2] == h:
                    learning_rate = learning_rate/2
                    session.run(tf.assign(m.lr, learning_rate))
                elif last3[1] == h:
                    if (last3[1] - last3[2])/last3[1] < tol:
                        learning_rate = learning_rate/2
                        session.run(tf.assign(m.lr, learning_rate))
                else:
                    if (h - min(last3))/h < tol:
                        learning_rate = learning_rate/2
                        session.run(tf.assign(m.lr, learning_rate))
                last3 = last3[1:] + [median_cost]
            elif len(last3) < 3:
                last3 = last3 + [median_cost]
            else:
                raise Exception





 
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gsd","--graph_sub_dir", help="Use this subdir in graphs for tensorboard. str not a real path.")
    parser.add_argument("-c", "--continue_training", help="""Set this to continue training rather than 
                                   start from scratch. -m is required to specify the
                                   starting model and -i for the starting saved model index""",
                              action="store_true", default=False)
    parser.add_argument("-m", "--saved_model_path", help="Path to saved model to start training")
    parser.add_argument("-i", "--starting_index", help="Number to start training/saving from", type=int)
    parser.add_argument("-d", "--debug", help="Set this for logging.DEBUG", action='store_true')
    args = parser.parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(level)
    graph_path = graph_path + "/" + args.graph_sub_dir if args.graph_sub_dir else graph_path
    if args.continue_training:
        assert args.saved_model_path and args.starting_index, "Wrong arguments see -h for details"
        main(graph_path, continue_training=True, start_model=args.saved_model_path, start_ind=args.starting_index)
    else:
        main(graph_path)



