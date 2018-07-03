import tensorflow as tf
import numpy as np

class Data:
    def __init__(self, FLAGS, train_source, train_target, test_source, test_target, vocabulary):
        self.FLAGS = FLAGS
        self.train_source = train_source
        self.train_target = train_target
        self.test_source = test_source
        self.test_target = test_target


        # create vocab and reverse vocab maps
        self.vocab     = {}
        self.rev_vocab = {}
        self.END_TOKEN = 1 
        self.UNK_TOKEN = 2
        with open(vocabulary, "r") as f:
            for idx, line in enumerate(f):
                self.vocab[line.strip()] = idx
                self.rev_vocab[idx] = line.strip()
        self.vocab_size = len(self.vocab)
        self.reference_corpus = []
        self.translation_corpus = []
        self.iteration = 0
        self.ak = []

    def builtTranslationCorpus(self, translations):
        corpus = []
        for token in translations:
            sentence = []
            for t in token:
                if(t != -1):
                    s = self.rev_vocab[t]
                    if(s != ' ' and s != '.' and s != ',' and s != '</S>' and s != '<S>'):
                        sentence.append(s)

            # corpus.append(' '.join(sentence).replace('</S>','').replace('<S>', ''))
            corpus.append(sentence)


        self.translation_corpus = corpus

    def tokenize_and_map(self,line):
        line = line.replace('\n', '')
        return [self.vocab.get(token, self.UNK_TOKEN) for token in line.split(' ')]

    def make_test_fn(self):

        def sampler():
            with open(self.test_source, "r") as finput, open(self.test_target, "r") as foutput:
                for source,target in zip(finput, foutput):
                    self.reference_corpus.append(target.rstrip().split(" "))     
                    yield {
                        'input': [0] + self.tokenize_and_map(source)[:self.FLAGS.input_max_length - 1] + [self.END_TOKEN],
                        'output': [0] + self.tokenize_and_map(target)[:self.FLAGS.output_max_length - 1] + [self.END_TOKEN]
                    }

        data_feed = sampler()

        def feed_fn():
            source, target = [], []
            input_length, output_length = 0, 0
            for rec in data_feed:
                # rec = next(data_feed)
                source.append(rec['input'])
                target.append(rec['output'])
                input_length = max(input_length, len(source[-1]))
                output_length = max(output_length, len(target[-1]))

            for i in range(0, len(source)):
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
                with open(self.train_source) as finput, open(self.train_target) as foutput:
                    for source,target in zip(finput, foutput):
                        yield {
                            'input': [0] + self.tokenize_and_map(source)[:self.FLAGS.input_max_length - 1] + [self.END_TOKEN],
                            'output': [0] + self.tokenize_and_map(target)[:self.FLAGS.output_max_length - 1] + [self.END_TOKEN]
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

