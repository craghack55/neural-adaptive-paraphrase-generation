import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense

class Seq2seq:
    def __init__(self, vocab_size, FLAGS, transferMethod, sourceCheckpointPath = None, loadParameters = True, inferGraph = 0):
        self.FLAGS = FLAGS
        self.vocab_size = vocab_size
        self.transferMethod = transferMethod
        self.sourceCheckpointPath = sourceCheckpointPath
        self.loadParameters = loadParameters
        self.inferGraph = inferGraph

    def setLoadParameters(self, loadParameters):
        self.loadParameters = loadParameters

    def make_graph(self,mode, features, labels, params):
        embed_dim = self.FLAGS.embed_dim
        num_units = self.FLAGS.num_units

        input, output   = features['input'], features['output']
        batch_size     = tf.shape(input)[0]
        start_tokens   = tf.zeros([batch_size], dtype= tf.int64)
        train_output   = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
        input_lengths  = tf.reduce_sum(tf.to_int32(tf.not_equal(input, 1)), 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
        input_embed    = layers.embed_sequence(input, vocab_size=self.vocab_size, embed_dim = embed_dim, scope = 'embed')
        output_embed   = layers.embed_sequence(train_output, vocab_size=self.vocab_size, embed_dim = embed_dim, scope = 'embed', reuse = True)
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
        cell2 = tf.contrib.rnn.LSTMCell(num_units=num_units)
        cell3 = tf.contrib.rnn.LSTMCell(num_units=num_units)


        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)
        cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)

        if self.FLAGS.use_residual_lstm:
            cell2 = tf.contrib.rnn.ResidualWrapper(cell2)

        if(self.transferMethod == "scheme3"):
            cell4 = tf.contrib.rnn.LSTMCell(num_units=num_units)
            cell4 = tf.nn.rnn_cell.DropoutWrapper(cell4, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)
            cells = [cell, cell2, cell3, cell4]

        else:
            cells = [cell, cell2, cell3]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(multi_rnn_cell, input_embed, dtype=tf.float32)
        # encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)

        
        if(self.transferMethod is not None and self.loadParameters):
            if(self.transferMethod == "embeddingOnly"):
                assignment_map = {
                    'embed/': 'embed/'
                }
            else:
                assignment_map = {
                    'rnn/multi_rnn_cell/cell_1/': 'rnn/multi_rnn_cell/cell_1/',
                    'rnn/multi_rnn_cell/cell_0/': 'rnn/multi_rnn_cell/cell_0/',
                    'rnn/multi_rnn_cell/cell_2/': 'rnn/multi_rnn_cell/cell_2/',
                    'embed/': 'embed/'
                }

            tf.train.init_from_checkpoint(self.sourceCheckpointPath, assignment_map)


        def decode(helper = None, mode = None, scope = None, reuse = None):
            # Decoder is partially based on @ilblackdragon//tf_example/seq2seq.py
            with tf.variable_scope(scope, reuse=reuse):

                cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
                cell2 = tf.contrib.rnn.LSTMCell(num_units=num_units)
                cell3 = tf.contrib.rnn.LSTMCell(num_units=num_units)

                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)
                cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)
                cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)

                if self.FLAGS.use_residual_lstm:
                    cell2 = tf.contrib.rnn.ResidualWrapper(cell2)


                if(self.transferMethod == "scheme3"):
                    cell4 = tf.contrib.rnn.LSTMCell(num_units=num_units)
                    cell4 = tf.nn.rnn_cell.DropoutWrapper(cell4, output_keep_prob = 1 - self.FLAGS.drop_prob, input_keep_prob = 1 - self.FLAGS.drop_prob)
                    cells = [cell, cell2, cell3, cell4]

                else:
                    cells = [cell, cell2, cell3]

                multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                
                if(self.transferMethod is not None and self.loadParameters):
                    if(self.transferMethod != "embeddingOnly"):

                        assignment_map = {
                            'decode/memory_layer/' : 'decode/memory_layer/',
                            'decode/decoder/output_projection_wrapper/attention_wrapper/multi_rnn_cell/cell_1/' : 'decode/decoder/output_projection_wrapper/attention_wrapper/multi_rnn_cell/cell_1/',
                            'decode/decoder/output_projection_wrapper/attention_wrapper/multi_rnn_cell/cell_2/' : 'decode/decoder/output_projection_wrapper/attention_wrapper/multi_rnn_cell/cell_2/',
                            'decode/decoder/output_projection_wrapper/attention_wrapper/multi_rnn_cell/cell_0/' : 'decode/decoder/output_projection_wrapper/attention_wrapper/multi_rnn_cell/cell_0/',
                            'decode/decoder/output_projection_wrapper/attention_wrapper/bahdanau_attention/' : 'decode/decoder/output_projection_wrapper/attention_wrapper/bahdanau_attention/',
                            'decode/decoder/output_projection_wrapper/kernel/' : 'decode/decoder/output_projection_wrapper/kernel/',
                            'decode/decoder/output_projection_wrapper/attention_wrapper/attention_layer/' : 'decode/decoder/output_projection_wrapper/attention_wrapper/attention_layer/',
                            'decode/decoder/output_projection_wrapper/bias/' : 'decode/decoder/output_projection_wrapper/bias/'
                        }

                        tf.train.init_from_checkpoint(self.sourceCheckpointPath, assignment_map)

                if(self.inferGraph == 1):

                    enc_rnn_out_beam   = tf.contrib.seq2seq.tile_batch(encoder_outputs   , 5)
                    seq_len_beam       = tf.contrib.seq2seq.tile_batch(input_lengths       , 5)
                    enc_rnn_state_beam = tf.contrib.seq2seq.tile_batch(encoder_final_state , 5)
                                                
                    # start tokens mean be the original batch size so divide
                    s_t = tf.tile(tf.to_int32(start_tokens), [batch_size] )
                        
                    attn_mech_beam = tf.contrib.seq2seq.BahdanauAttention(num_units = num_units,  memory = enc_rnn_out_beam,  memory_sequence_length = seq_len_beam)

                    cell_beam = tf.contrib.seq2seq.AttentionWrapper(cell = multi_rnn_cell, attention_mechanism = attn_mech_beam, attention_layer_size = num_units / 2)  

                    out_cell = tf.contrib.rnn.OutputProjectionWrapper(cell_beam, self.vocab_size, reuse=reuse)

                    initial_state_beam = out_cell.zero_state(batch_size=batch_size * 5, dtype=tf.float32).clone(cell_state = enc_rnn_state_beam)



                    decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = out_cell,
                                                                        embedding = embeddings,
                                                                        start_tokens = tf.to_int32(start_tokens),
                                                                        end_token = 1,
                                                                        initial_state = initial_state_beam,
                                                                        beam_width = 5,
                                                                        length_penalty_weight = 0.0)
                    outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=False, maximum_iterations=self.FLAGS.output_max_length)

                    return outputs[0].predicted_ids
                                
                else:

                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = num_units, memory = encoder_outputs, memory_sequence_length = input_lengths)

                    attn_cell = tf.contrib.seq2seq.AttentionWrapper(multi_rnn_cell, attention_mechanism, attention_layer_size = num_units / 2)
                    out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, self.vocab_size, reuse = reuse)
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))

                    outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder, output_time_major=False,
                        impute_finished=True, maximum_iterations=self.FLAGS.output_max_length)

                    return outputs[0]

        if(mode == tf.contrib.learn.ModeKeys.INFER):
            if(inferGraph == 1):
                pred_outputs = decode(None, mode, 'decode')

                # tf.identity(pred_outputs[0], name='predict')

                return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs)

            else:
                pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
                pred_outputs = decode(pred_helper, mode, 'decode')

                tf.identity(pred_outputs.sample_id[0], name='predict')

                return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.sample_id)
        else:
            train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token = 1)
            train_outputs = decode(train_helper, mode, 'decode')
            pred_outputs = decode(pred_helper, mode, 'decode', reuse=True)

            tf.identity(train_outputs.sample_id[0], name='train_pred')
            weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
            loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, output, weights=weights)

            tvars = tf.trainable_variables()

            for i in tvars:
                print(i)
            
            if(self.transferMethod == "schema2"):
                train_vars = [var for var in tvars if not "cell_0" or not "cell_1" in var.name]
            else:
                if(self.transferMethod == "schema3"):
                    train_vars = [var for var in tvars if not "cell_0" or not "cell_1" or not "cell_2" in var.name]
                else:
                    if(self.transferMethod == "schema1"):
                        train_vars = [var for var in tvars if not "cell_0" in var.name]
                    else:
                        train_vars = tvars

            train_op = layers.optimize_loss(
                loss, tf.train.get_global_step(),
                optimizer=self.FLAGS.optimizer,
                learning_rate=self.FLAGS.learning_rate,
                summaries=['loss', 'learning_rate'],
                variables = train_vars)

            tf.identity(pred_outputs.sample_id[0], name='predict')
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.sample_id, loss=loss, train_op=train_op)

