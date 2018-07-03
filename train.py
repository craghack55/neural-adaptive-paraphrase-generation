import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from seq2seq import Seq2seq
from data_handler import Data
from tensor2tensor.utils import bleu_hook
from tensorflow.contrib.framework.python.framework import checkpoint_utils

flags = tf.flags 
FLAGS = tf.flags.FLAGS

# Model related
flags.DEFINE_integer('num_units'         , 512           , 'Number of units in a LSTM cell')
flags.DEFINE_integer('embed_dim'         , 50           , 'Size of the embedding vector')
flags.DEFINE_integer('length_penalty_weight'         , 0           , '')
flags.DEFINE_integer('beam_width'         , 5           , '')
flags.DEFINE_float('drop_prob'         , 0.5           , '')

# Training related
flags.DEFINE_float('learning_rate'       , 0.001         , 'learning rate for the optimizer')
flags.DEFINE_string('optimizer'          , 'Adam'        , 'Name of the train source file')
flags.DEFINE_integer('batch_size'        , 50            , 'random seed for training sampling')
flags.DEFINE_integer('print_every' , 10 				, 	'print records every n iteration') 
flags.DEFINE_integer('iterations' , 10000 				, 'number of iterations to train')
flags.DEFINE_string('model_dir'          		, 'checkpoints' , 'Directory where to save the model')

flags.DEFINE_integer('input_max_length'  , 100            , 'Max length of input sequence to use')
flags.DEFINE_integer('output_max_length' , 100            , 'Max length of output sequence to use')

flags.DEFINE_bool('use_residual_lstm'    , True          , 'To use the residual connection with the residual LSTM')

# Data related
flags.DEFINE_string('input_filename', 'data/mscoco/train_source.txt', 'Name of the train source file')
flags.DEFINE_string('output_filename', 'data/mscoco/train_target.txt', 'Name of the train target file')
flags.DEFINE_string('input_test_filename', 'data/mscoco/test_source.txt', 'Name of the train source file')
flags.DEFINE_string('output_test_filename', 'data/mscoco/test_target.txt', 'Name of the train target file')
flags.DEFINE_string('vocab_filename', 'data/mscoco/train_vocab.txt', 'Name of the vocab file')


def saveResult(percentage, score, resultPath):
    file = open(resultPath + str(percentage) + ".txt","w")
    file.write(str(percentage) + " " + str(score))
    file.close()


# MSCOCO - 167149 data points.

def evaluate(reference_corpus, translation_corpus):
    bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
    return bleu


def trainWithPreviousKnowledge(datasetPath, datasetSize, transferMethod = None, transferVocabularyPath = None, sourceCheckpointPath = None, suffix = ""):
    percentages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    checkpoint_filename = "checkpointsAdaptiveWithPrev" + suffix
    size = datasetSize
    epoch = 5
    test_source = datasetPath + "test_source.txt"
    test_target = datasetPath + "test_target.txt"
    resultPath = datasetPath + "AdaptiveWithPrev"

    if(transferMethod is None):
        vocabulary = datasetPath + "quora_msr_vocabulary.txt"
    else:
        vocabulary = transferVocabularyPath

    data  = Data(FLAGS, "", "", "", "", vocabulary)
    model = Seq2seq(data.vocab_size, FLAGS, transferMethod, sourceCheckpointPath)

    for i in percentages:
        source_filename = datasetPath + format(i, '.1f') + "_source" + ".txt"
        target_filename = datasetPath + format(i, '.1f') + "_target" + ".txt"
        data  = Data(FLAGS, source_filename, target_filename, test_source, test_target, vocabulary)
        iterations = int(round(size * i * epoch / FLAGS.batch_size))
        # iterations = 1

        input_fn, feed_fn = data.make_input_fn()
        test_fn = data.make_test_fn()
        print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
                formatter=data.get_formatter(['source', 'target', 'predict']))

        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpoint_filename, params=FLAGS)
        print("Training with " + format(i, '.2f') + " percent of the dataset.")
        estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

        model.setLoadParameters(False)

        # test_paraphrases = list(estimator.predict(test_fn))

        a = estimatorInfer.predict(test_fn)

        test_paraphrases = []

        # a = list(a)

        # for i in a:
        #     print(i.shape)

        # print(len(a))

        for i in a:
            j = i[:, 0]
            test_paraphrases.append(j)

        data.builtTranslationCorpus(test_paraphrases)
        scr = evaluate(data.reference_corpus, data.translation_corpus)
        print(i, scr)
        saveResult(i, scr, resultPath)

# Restore from checkpoint. Resume training with different dataset.
def trainWithPreviousKnowledgePool(datasetPath, datasetSize, transferMethod = None, transferVocabularyPath = None, sourceCheckpointPath = None, suffix = ""):
    percentages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    checkpoint_filename = "checkpointsAdaptiveWithPrevPool" + suffix
    size = datasetSize
    epoch = 5
    test_source = datasetPath + "test_source.txt"
    test_target = datasetPath + "test_target.txt"
    resultPath = datasetPath + "AdaptiveWithPrevWithPool"

    if(transferMethod is None):
        vocabulary = datasetPath + "quora_msr_vocabulary.txt"
    else:
        vocabulary = transferVocabularyPath

    data  = Data(FLAGS, "", "", "", "", vocabulary)
    model = Seq2seq(data.vocab_size, FLAGS, transferMethod, sourceCheckpointPath)


    for i in percentages:
        source_filename = datasetPath + format(i, '.1f') + "_pool_source" + ".txt"
        target_filename = datasetPath + format(i, '.1f') + "_pool_target" + ".txt"
        data  = Data(FLAGS, source_filename, target_filename, test_source, test_target, vocabulary)
        iterations = int(round(size * i * epoch / FLAGS.batch_size))
        # iterations = 1

        input_fn, feed_fn = data.make_input_fn()
        test_fn = data.make_test_fn()
        print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
                formatter=data.get_formatter(['source', 'target', 'predict']))

        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpoint_filename, params=FLAGS)
        print("Training with " + format(i, '.2f') + " percent of the dataset.")
        estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

        model.setLoadParameters(False)

        # test_paraphrases = list(estimator.predict(test_fn))

        a = estimatorInfer.predict(test_fn)

        test_paraphrases = []

        # a = list(a)

        # for i in a:
        #     print(i.shape)

        # print(len(a))

        for i in a:
            j = i[:, 0]
            test_paraphrases.append(j)

        data.builtTranslationCorpus(test_paraphrases)
        scr = evaluate(data.reference_corpus, data.translation_corpus)
        print(i, scr)
        saveResult(i, scr, resultPath)


# Keep checkpoint folders seperate. Restart training with each different block.
def trainWithoutPreviousKnowledge(datasetPath, datasetSize, transferMethod = None, transferVocabularyPath = None, sourceCheckpointPath = None, suffix = ""):
    percentages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    size = datasetSize
    epoch = 3
    test_source = datasetPath + "test_source.txt"
    test_target = datasetPath + "test_target.txt"
    resultPath = datasetPath + "AdaptiveWithoutPrev"

    if(transferMethod is None):
        vocabulary = datasetPath + "quora_msr_vocabulary.txt"
    else:
        vocabulary = transferVocabularyPath

	
    for i in percentages:
        source_filename = datasetPath + format(i, '.1f') + "_pool_source" + ".txt"
        target_filename = datasetPath + format(i, '.1f') + "_pool_target" + ".txt"
        checkpoint_filename = "checkpointsWithoutPrev" + format(i, '.1f') + suffix
        data  = Data(FLAGS, source_filename, target_filename, test_source, test_target, vocabulary)
        model = Seq2seq(data.vocab_size, FLAGS, transferMethod, sourceCheckpointPath)
        iterations = int(round(size * i * epoch / FLAGS.batch_size))
        # iterations = 1

        input_fn, feed_fn = data.make_input_fn()
        print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
                formatter=data.get_formatter(['source', 'target', 'predict']))

        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpoint_filename, params=FLAGS)
        print("Training with " + format(i, '.2f') + " percent of the dataset.")
        estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

        model.setLoadParameters(False)
        test_fn = data.make_test_fn()
        # test_paraphrases = list(estimator.predict(test_fn))

        a = estimatorInfer.predict(test_fn)

        test_paraphrases = []

        # a = list(a)

        # for i in a:
        #     print(i.shape)

        # print(len(a))

        for i in a:
            j = i[:, 0]
            test_paraphrases.append(j)

        data.builtTranslationCorpus(test_paraphrases)
        print(data.translation_corpus)
        scr = evaluate(data.reference_corpus, data.translation_corpus)
        print(i, scr)
        saveResult(i, scr, resultPath)

def trainWithActiveLearning(samplingMethod):
	print("")

def supervisedLearning(datasetPath, datasetSize, transferMethod = None, transferVocabularyPath = None, sourceCheckpointPath = None, suffix = ""):

    test_source = datasetPath + "test_source.txt"
    test_target = datasetPath + "test_target.txt"
    train_source = datasetPath + "train_source.txt"
    train_target = datasetPath + "train_target.txt"
    resultPath = datasetPath + "SL"

    if(transferMethod is None):
        vocabulary = datasetPath + "quora_msr_vocabulary.txt"
    else:
        vocabulary = transferVocabularyPath


    data  = Data(FLAGS, train_source, train_target, test_source, test_target, vocabulary)
    model = Seq2seq(data.vocab_size, FLAGS, transferMethod, sourceCheckpointPath)
    size = datasetSize
    epoch = 5
    # iterations = int(round(size * epoch / FLAGS.batch_size))
    iterations = 1

    # var_list = checkpoint_utils.list_variables('checkpoints')
    # for v in var_list: print(v)

    input_fn, feed_fn = data.make_input_fn()
    print_inputs = tf.train.LoggingTensorHook(['source', 'target', 'predict'], every_n_iter=FLAGS.print_every,
            formatter=data.get_formatter(['source', 'target', 'predict']))

    estimator = tf.estimator.Estimator(model_fn = model.make_graph, model_dir="checkpointsQuoraMSR")
    estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=iterations)

    # modelInfer = Seq2seq(data.vocab_size, FLAGS, transferMethod, sourceCheckpointPath, False, inferGraph = 1)    
    # estimatorInfer = tf.estimator.Estimator(model_fn = modelInfer.make_graph, model_dir="cps/checkpointsQuoraMSR3")

    test_fn = data.make_test_fn()


    a = estimator.predict(test_fn)

    test_paraphrases = []

    # a = list(a)

    # for i in a:
    #     print(i.shape)

    # print(len(a))

    for i in a:
        j = i[:, 0]
        test_paraphrases.append(j)

    data.builtTranslationCorpus(test_paraphrases)
    scr = evaluate(data.reference_corpus, data.translation_corpus)
    print(data.translation_corpus)
    print(scr)
    saveResult(100, scr, resultPath)


def main(argv):
    # tf.logging._logger.setLevel(logging.INFO)
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    datasetPath = 'data/msr/'
    #datasetPath = 'data/quora/'
    # datasetPath = 'data/ppdb-lexical/'
    msrSize = 2753
    quoraSize = 119445
    ppdbSize = 231716
    # train_source = 'data/quora/'

    # msrSize = 2753


    # QUORA
    # SL = 0.022275709 - 2.22
    # 1 = 0.10044589 - 10.04
    # 2 = 0.0874715 - 8.74
    # 3 = 0.083236344 - 8.32 
    # INIT = 0.08980903 - 8.9
    # Embedding = 0.023836417 - 2.38

    # PPDB
    # SL = 0.022275709 - 2.22
    # 1 = 0.017379494 = 1.73
    # 2 = 0.024661293 = 2.46
    # 3 = 0.023211326 = 2.32
    # INIT = 0.030824652 = 3.08 
    # Embedding = 0.0073982994 - 0.73

    # Without previous
    # 0.2 - 0.0012960344 0.12
    # 0.3 - 0.0027647084 0.27
    # 0.4 - 0.006054903 0.6
    # 0.5 - 0.013206709 1.32
    # 0.6 - 0.009628898 0.96
    # 0.7 - 0.020879177 1.08
    # 0.8 - 0.012489422 1.24
    # 0.9 - 0.0126661025 1.26
    # All - 0.01347382 1.34

    # With previous
    # 0.2 0.0010355313 0.1
    # 0.3 0.009242473 0.92
    # 0.4 0.009493898 0.94
    # 0.5 0.015665574 1.56
    # 0.6 0.011380633 1.13
    # 0.7 0.011143916 1.11
    # 0.8 0.012659637 1.26
    # 0.9 0.013756248 1.37
    # All 0.014124637 1.41

    # With previous without Pooling
    # 0.2 0.00027219247 0.027
    # 0.3 0.0005523451 0.055
    # 0.4 0.0021470916 0.21
    # 0.5 0.004706152 0.47
    # 0.6 0.0077507854 0.77
    # 0.7 0.009900585 0.99
    # 0.8 0.010409241 1.04
    # 0.9 0.011943593 1.19

    # train_source = 'data/ppdb-lexical/'
    # vocab = 'data/ppdb-lexical/'
    # msrSize = 2753


    # trainWithoutPreviousKnowledge(datasetPath, msrSize)

    # trainWithPreviousKnowledge(datasetPath, msrSize, "scheme3", datasetPath + "quora_msr_vocabulary.txt", "checkpoints", suffix = "scheme3")
    # trainWithPreviousKnowledgePool(datasetPath, msrSize, "scheme3", datasetPath + "quora_msr_vocabulary.txt", "checkpoints", suffix = "scheme3")
    # trainWithoutPreviousKnowledge(datasetPath, msrSize, "scheme3", datasetPath + "quora_msr_vocabulary.txt", "checkpoints", suffix = "scheme3")
    # supervisedLearning(datasetPath, msrSize, "scheme3", datasetPath + "quora_msr_vocabulary.txt", "checkpoints", suffix = "scheme3")
    supervisedLearning(datasetPath, msrSize)


if __name__ == "__main__":
    tf.app.run()
