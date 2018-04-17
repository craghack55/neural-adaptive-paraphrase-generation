import tensorflow as tf
import numpy as np

class Data:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        # create vocab and reverse vocab maps
        self.vocab     = {}
        self.rev_vocab = {}
        self.END_TOKEN = 1 
        self.UNK_TOKEN = 2
        with open(FLAGS.vocab_filename) as f:
            for idx, line in enumerate(f):
                self.vocab[line.strip()] = idx
                self.rev_vocab[idx] = line.strip()
        self.vocab_size = len(self.vocab)
        self.reference_corpus = []
        self.translation_corpus = []

    def builtTranslationCorpus(self, translations):
        corpus = []
        for token in translations:
            sentence = []
            for t in token:
                sentence.append(self.rev_vocab[t])
            corpus.append(' '.join(sentence).replace('</S>','').replace('<S>', ''))

        self.translation_corpus = corpus

    def tokenize_and_map(self,line):
        line = line.replace('\n', '')
        return [self.vocab.get(token, self.UNK_TOKEN) for token in line.split(' ')]

    def make_test_fn(self):

        def sampler():
            while True:
                with open(self.FLAGS.input_test_filename) as finput, open(self.FLAGS.output_test_filename) as foutput:
                    for source,target in zip(finput, foutput):
                        self.reference_corpus.append(target.rstrip())     
                        yield {
                            'input': self.tokenize_and_map(source)[:self.FLAGS.input_max_length - 1] + [self.END_TOKEN],
                            'output': self.tokenize_and_map(target)[:self.FLAGS.output_max_length - 1] + [self.END_TOKEN]
                        }

        data_feed = sampler()

        def feed_fn():
            source, target = [], []
            input_length, output_length = 0, 0
            for i in range(self.FLAGS.batch_size):
                rec = next(data_feed)
                source.append(rec['input'])
                target.append(rec['output'])
                input_length = max(input_length, len(source[-1]))
                output_length = max(output_length, len(target[-1]))

            for i in range(self.FLAGS.batch_size):
                source[i] += [self.END_TOKEN] * (input_length - len(source[i]))
                target[i] += [self.END_TOKEN] * (output_length - len(target[i]))

            return { 'input': np.asarray(source), 'output': np.asarray(target)}

        f = feed_fn()

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=f, num_epochs=1, shuffle=False)

        return predict_input_fn
         

    def make_input_fn(self):

        def input_fn():
            inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
            output = tf.placeholder(tf.int64, shape=[None, None], name='output')
            tf.identity(inp[0], 'source')
            tf.identity(output[0], 'target')
            return { 'input': inp, 'output': output}, None

        def sampler():
            while True:
                with open(self.FLAGS.input_filename) as finput, open(self.FLAGS.output_filename) as foutput:
                    for source,target in zip(finput, foutput):
                        yield {
                            'input': self.tokenize_and_map(source)[:self.FLAGS.input_max_length - 1] + [self.END_TOKEN],
                            'output': self.tokenize_and_map(target)[:self.FLAGS.output_max_length - 1] + [self.END_TOKEN]
                        }

        data_feed = sampler()

        def feed_fn():
            source, target = [], []
            input_length, output_length = 0, 0
            for i in range(self.FLAGS.batch_size):
                rec = next(data_feed)
                source.append(rec['input'])
                target.append(rec['output'])
                input_length = max(input_length, len(source[-1]))
                output_length = max(output_length, len(target[-1]))

            for i in range(self.FLAGS.batch_size):
                source[i] += [self.END_TOKEN] * (input_length - len(source[i]))
                target[i] += [self.END_TOKEN] * (output_length - len(target[i]))
            return { 'input:0': source, 'output:0': target }

        return input_fn, feed_fn

    def get_formatter(self,keys):
        def to_str(sequence):
            tokens = [
                self.rev_vocab.get(x, "<UNK>") for x in sequence]
            return ' '.join(tokens)

        def format(values):
            res = []
            for key in keys:
                res.append("****%s == %s" % (key, to_str(values[key]).replace('</S>','').replace('<S>', '')))
            return '\n'+'\n'.join(res)
        return format

