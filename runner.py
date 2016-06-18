import os
import tensorflow as tf
import numpy as np
import pickle
import sys
import argparse
import logging
from model import RNNModel, CNNModel, RNNRNNModel
from reader import TokReader, CharReader, CharTokReader
import logging
logger = logging.getLogger("USF")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]: %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)
graph_path = 'graphs/'
model_dir = "models/"
try:
    with open('tok_map.pkl', 'rb') as f:
        tok_map = pickle.load(f)
    with open('char_map.pkl', 'rb') as f: 
        char_map = pickle.load(f)
except FileNotFoundError:
    print("tok_map.pkl and char_map.pkl mappings not found. Please run preprocess.py")
    sys.exit()
assert tok_map["*PAD*"] == 0, "The token mapping must contain *PAD* as index 0"
assert tok_map["*UNK*"] == 1, "The token mapping must contain *UNK* as the index 1"
assert char_map["*PAD*"] == 0, "The token mapping must contain *PAD* as index 0"
assert char_map["*UNK*"] == 1, "The token mapping must contain *UNK* as the index 1"
assert char_map["*START*"] == 2, "The char mapping must contain *START* as the index 2"
assert char_map["*END*"] == 3, "The char mapping must contain *END* as the index 3"


def main(graph_path, Model, stream, validstream, continue_training=False, 
        start_model=None, start_ind=0, save_every=1):
    """Run a complete training session. Will load a saved model to continue training
    if provided. After every epoch the current model will be saved, and the tensorboard
    will graph new data.
    """  
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-Config.init_scale,
                                                     Config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(config=Config)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(max_to_keep=Config.num_models)
        if continue_training:
            print("Continuing training from saved model ",start_model)
            saver.restore(session,start_model)
        writer = tf.train.SummaryWriter(graph_path, max_queue=3) 
        last3 = []
        learning_rate = Config.learning_rate
        session.run(tf.assign(m.lr, learning_rate))
        tol = 0.001
        for i in range(start_ind, start_ind+Config.num_epochs):
            print("EPOCH: %s"%i)
            print("learning_rate: %s"%learning_rate)
            epoch_cost, median_cost, max_cost = m.run_epoch(session, stream.get_sents(), True)   
            print("Total cost for EPOCH: %s"%i)
            print(epoch_cost)
            print("Median cost: %s"%median_cost)
            print("Max cost: %s"%max_cost)
            accuracy = m.run_epoch(session, validstream.get_sents(), False)
            print("accuracy: %s"%accuracy)
            summ1 = tf.scalar_summary("epoch_cost", tf.constant(epoch_cost))
            summ2 = tf.scalar_summary("median_cost", tf.constant(median_cost))
            summ3 = tf.scalar_summary("max_cost", tf.constant(max_cost))
            summ4 = tf.scalar_summary("learning_rate", tf.constant(learning_rate))
            summ5 = tf.scalar_summary("accuracy", tf.constant(accuracy))
            merge = tf.merge_summary([summ1, summ2, summ3, summ4, summ5])
            writer.add_summary(merge.eval(), i)
            if i % save_every == 0:
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
    parser.add_argument("model", help="""Which model to use, required. Options are tokrnn, charrnn, tokconv, charconv""")
    args = parser.parse_args()
    debug = args.debug
    
    if debug:
        level = logging.DEBUG
        limit = 500
    else:
        level = logging.INFO
        limit = None
    logging.getLogger().setLevel(level)
    
    graph_path = graph_path + "/" + args.graph_sub_dir if args.graph_sub_dir else graph_path
    
    if args.model == "tokrnn":
        from config import TokRNNConfig as Config
        Config.vocab_size = len(tok_map)
        Config.sent_len = 10 if debug else Config.sent_len
        Config.batch_size = 10 if debug else Config.batch_size
        stream = TokReader(Config.sent_len, Config.batch_size, tok_map, random=True, 
                           rounded=True, training=True, limit=limit)
        validstream = TokReader(Config.sent_len, Config.batch_size, tok_map, random=True, 
                                rounded=True, training=False, limit=limit)
        Model = RNNModel
    elif args.model == "charrnn":
        from config import CharRNNConfig as Config
        Config.vocab_size = len(char_map)
        Config.sent_len =  10 if debug else Config.sent_len
        Config.batch_size = 10 if debug else Config.batch_size
        stream = CharReader(Config.sent_len, Config.batch_size, char_map, random=True, 
                            rounded=True, training=True, limit=limit)
        validstream = CharReader(Config.sent_len, Config.batch_size, char_map, random=True, 
                            rounded=True, training=False, limit=limit)
        Model = RNNModel
    elif args.model == "tokconv":
        from config import TokConvConfig as Config
        Config.vocab_size = len(tok_map)
        Config.sent_len = 10 if debug else Config.sent_len
        Config.batch_size = 10 if debug else Config.batch_size
        stream = TokReader(Config.sent_len, Config.batch_size, tok_map, random=True, 
                           rounded=True, training=True, limit=limit)
        validstream = TokReader(Config.sent_len, Config.batch_size, tok_map, random=True, 
                                rounded=True, training=False, limit=limit)
        Model = CNNModel
    elif args.model == "charconv":
        from config import CharConvConfig as Config
        Config.vocab_size = len(char_map)
        Config.sent_len = 10 if debug else Config.sent_len
        Config.batch_size = 10 if debug else Config.batch_size
        stream = TokReader(Config.sent_len, Config.batch_size, char_map, random=True, 
                           rounded=True, training=True, limit=limit)
        validstream = TokReader(Config.sent_len, Config.batch_size, char_map, random=True, 
                                rounded=True, training=False, limit=limit)
        Model = CNNModel
    elif args.model == "chartokrnn":
        from config import CharTokRNNConfig as Config
        Config.vocab_size = len(char_map)
        Config.sent_len = 10 if debug else Config.sent_len
        Config.word_len = 10 if debug else Config.word_len
        Config.batch_size = 10 if debug else Config.batch_size
        stream = CharTokReader(Config.sent_len, Config.word_len, Config.batch_size, 
                               char_map, random=True, rounded=True, training=True, limit=limit)
        validstream = CharTokReader(Config.sent_len, Config.word_len, Config.batch_size, 
                                    char_map, random=True, rounded=True, training=False, limit=limit)
        Model = RNNRNNModel
    else:
        raise NotImplementedError("See -h for details on which modes are supported")

    if args.continue_training:
        assert args.saved_model_path and args.starting_index, "Wrong arguments see -h for details"
        main(graph_path, Model, stream, validstream,
             continue_training=True, start_model=args.saved_model_path, 
             start_ind=args.starting_index, save_every=Config.save_every)
    else:
        main(graph_path, Model, stream, validstream, save_every=Config.save_every)



