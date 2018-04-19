import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from seq2seq import Seq2seq
from data_handler import Data
from tensor2tensor.utils import bleu_hook

FLAGS = tf.flags.FLAGS

# Model related
tf.flags.DEFINE_integer('num_units'         , 256           , 'Number of units in a LSTM cell')
tf.flags.DEFINE_integer('embed_dim'         , 256           , 'Size of the embedding vector')
tf.flags.DEFINE_integer('length_penalty_weight'         , 0           , '')
tf.flags.DEFINE_integer('beam_width'         , 5           , '')


# Training related
tf.flags.DEFINE_float('learning_rate'       , 0.001         , 'learning rate for the optimizer')
tf.flags.DEFINE_string('optimizer'          , 'Adam'        , 'Name of the train source file')
tf.flags.DEFINE_integer('batch_size'        , 50            , 'random seed for training sampling')
tf.flags.DEFINE_integer('print_every' , 1000 				, 	'print records every n iteration') 
tf.flags.DEFINE_integer('iterations' , 10000 				, 'number of iterations to train')
tf.flags.DEFINE_string('model_dir'          		, 'checkpoints' , 'Directory where to save the model')

tf.flags.DEFINE_integer('input_max_length'  , 30            , 'Max length of input sequence to use')
tf.flags.DEFINE_integer('output_max_length' , 30            , 'Max length of output sequence to use')

tf.flags.DEFINE_bool('use_residual_lstm'    , True          , 'To use the residual connection with the residual LSTM')

# Data related
tf.flags.DEFINE_string('input_filename', 'data/mscoco/train_source.txt', 'Name of the train source file')
tf.flags.DEFINE_string('output_filename', 'data/mscoco/train_target.txt', 'Name of the train target file')
tf.flags.DEFINE_string('input_test_filename', 'data/mscoco/test_source.txt', 'Name of the train source file')
tf.flags.DEFINE_string('output_test_filename', 'data/mscoco/test_target.txt', 'Name of the train target file')
tf.flags.DEFINE_string('vocab_filename', 'data/mscoco/train_vocab.txt', 'Name of the vocab file')


def saveResult(percentage, score):
    file = open(str(percentage) + ".txt","w")
    file.write(str(percentage) + " " + str(score))
    file.close()


# MSCOCO - 167149 data points.

def evaluate(reference_corpus, translation_corpus):
    bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
    return bleu

# Restore from checkpoint. Resume training with different dataset.
def trainWithPreviousKnowledge(test_source, test_target, vocabulary):
    percentages = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    checkpoint_filename = "checkpointsAdaptiveWithPrev"
    size = 167479
    epoch = 3

    for i in percentages:
        source_filename = "data/mscoco/train_source" + format(i, '.2f') + ".txt"
        target_filename = "data/mscoco/train_target" + format(i, '.2f') + ".txt"
        data  = Data(FLAGS, source_filename, target_filename, test_source, test_target, vocabulary)
        model = Seq2seq(data.vocab_size, FLAGS)
        iterations = int(round(size * i * epoch / FLAGS.batch_size))

        input_fn, feed_fn = data.make_input_fn()
        test_fn = data.make_test_fn()
        print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
                formatter=data.get_formatter(['source', 'target', 'predict']))

        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpoint_filename, params=FLAGS)
        print("Training with " + format(i, '.2f') + " percent of the dataset.")
        estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

        test_paraphrases = list(estimator.predict(test_fn))
        data.builtTranslationCorpus(test_paraphrases)
        scr = evaluate(data.reference_corpus, data.translation_corpus)
        print(i, scr)
        saveResult(i, scr)


# Keep checkpoint folders seperate. Restart training with each different block.
def trainWithoutPreviousKnowledge(test_source, test_target, vocabulary):
    percentages = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    size = 167479
    epoch = 3
	
    for i in percentages:
        print("Training with " + str(i) + " percent of the dataset.")
        source_filename = "data/mscoco/train_source" + format(i, '.2f') + ".txt"
        target_filename = "data/mscoco/train_target" + format(i, '.2f') + ".txt"
        checkpoint_filename = "checkpoints" + format(i, '.2f')
        data  = Data(FLAGS, source_filename, target_filename, test_source, test_target, vocabulary)
        model = Seq2seq(data.vocab_size, FLAGS)
        iterations = int(round(size * i * epoch / FLAGS.batch_size))

        input_fn, feed_fn = data.make_input_fn()
        test_fn = data.make_test_fn()
        print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
                formatter=data.get_formatter(['source', 'target', 'predict']))

        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpoint_filename, params=FLAGS)
        print("Training with " + format(i, '.2f') + " percent of the dataset.")
        estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

        test_paraphrases = list(estimator.predict(test_fn))
        data.builtTranslationCorpus(test_paraphrases)
        scr = evaluate(data.reference_corpus, data.translation_corpus)
        print(i, scr)
        saveResult(i, scr)

# Train with MSCOCO. Restore from the checkpoint. Start training with Quora.
def trainWithTransferLearning(sourceDataset, transferMethod):

    mscoco_train_source = "data/" + sourceDataset + "/train_source.txt"
    mscoco_train_target = "data/" + sourceDataset + "/train_target.txt"
    mscoco_vocabulary = "data/" + sourceDataset + "/train_vocab.txt"
    quora_train_source = 'data/quora/train_source.txt'
    quora_train_target = 'data/quora/train_target.txt'
    quora_test_source = 'data/quora/test_source.txt'
    quora_test_target = 'data/quora/test_target.txt'
    quora_vocabulary = 'data/quora/train_vocab.txt'
    sourceDatasetIterations = 1
    targetDatasetIterations = 1
    model_directory = "checkpointsTransfer"


    # Train with MSCOCO dataset first  
    data  = Data(FLAGS, mscoco_train_source, mscoco_train_target, FLAGS.input_test_filename, FLAGS.output_test_filename, mscoco_vocabulary)
    model = Seq2seq(data.vocab_size, FLAGS)

    input_fn, feed_fn = data.make_input_fn()
    print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
            formatter=data.get_formatter(['source', 'target', 'predict']))

    estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpointsTransfer, params=FLAGS)
    estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=sourceDatasetIterations)

    # Load the model trained on MSCOCO and use it on Quora dataset.
    data  = Data(FLAGS, quora_train_source, quora_train_target, quora_test_source, quora_test_target, quora_vocabulary)

    input_fn, feed_fn = data.make_input_fn()
    test_fn = data.make_test_fn()
    print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
            formatter=data.get_formatter(['source', 'target', 'predict']))


    estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=targetDatasetIterations)

    test_paraphrases = list(estimator.predict(test_fn))
    data.builtTranslationCorpus(test_paraphrases)
    print(data.translation_corpus)
    print(evaluate(data.reference_corpus, data.translation_corpus))

def trainWithActiveLearning(samplingMethod):
	print("")

def supervisedLearning(train_source, train_target, test_source, test_target, vocabulary):
    data  = Data(FLAGS, train_source, train_target, test_source, test_target, vocabulary)
    model = Seq2seq(data.vocab_size, FLAGS)
    iterations = 1

    input_fn, feed_fn = data.make_input_fn()
    test_fn = data.make_test_fn()
    print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
            formatter=data.get_formatter(['source', 'target', 'predict']))

    estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=FLAGS.model_dir, params=FLAGS)
    estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

    test_paraphrases = list(estimator.predict(test_fn))

    data.builtTranslationCorpus(test_paraphrases)
    print(data.reference_corpus)
    # print(data.translation_corpus)
    print(evaluate(data.reference_corpus, data.translation_corpus))


def main(argv):
    tf.logging._logger.setLevel(logging.INFO)

    train_source = 'data/mscoco/train_source.txt'
    train_target = 'data/mscoco/train_target.txt'
    test_source = 'data/mscoco/test_source.txt'
    test_target = 'data/mscoco/test_target.txt'
    vocab = 'data/mscoco/train_vocab.txt'


    trainWithPreviousKnowledge(test_source, test_target, vocab)
    # trainWithoutPreviousKnowledge(test_source, test_target, vocab)
    # trainWithTransferLearning("mscoco", "")
    # supervisedLearning(train_source, train_target, test_source, test_target, vocab)


if __name__ == "__main__":
    tf.app.run()
