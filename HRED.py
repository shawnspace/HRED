"impl for HRED model"

import tensorflow as tf
import modekeys
from tensorflow.python.layers import core as layers_core
import helper
from tensorflow.contrib.seq2seq.python.ops.helper import GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import tile_batch
from tensorflow.contrib.seq2seq.python.ops.decoder import dynamic_decode
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder

random_seed = 17

def impl(input_features,mode,hp):
    contexts = input_features['contexts'] # batch_size,max_con_length(with query),max_sen_length
    context_utterance_length = input_features['context_utterance_length'] # batch_size,max_con_length
    context_length = input_features['context_length'] #batch_size
    if mode == modekeys.TRAIN or mode == modekeys.EVAL:
        response_in = input_features['response_in']  # batch,max_res_sen
        response_out = input_features['response_out']  # batch,max_res_sen
        response_mask = input_features['response_mask']  # batch,max_res_sen, tf.float32
        batch_size = hp.batch_size
    else:
        batch_size = context_utterance_length.shape[0].value

    with tf.variable_scope('embedding_layer',reuse=tf.AUTO_REUSE) as vs:
        embedding_w = get_embedding_matrix(hp.word_dim,mode,hp.vocab_size,hp.word_embed_path,hp.vocab_path)
        contexts = tf.nn.embedding_lookup(embedding_w,contexts,'context_embedding')
        if mode == modekeys.TRAIN or mode == modekeys.EVAL:
            response_in = tf.nn.embedding_lookup(embedding_w, response_in, 'response_in_embedding')

    with tf.variable_scope('word_encoder_layer',reuse=tf.AUTO_REUSE) as vs:
        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=random_seed+1)
        bias_initializer = tf.zeros_initializer()
        fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.word_rnn_num_units, kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer)
        bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.word_rnn_num_units, kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer)

        context_t = tf.transpose(contexts, perm=[1, 0, 2, 3])  # max_con_length(with query),batch_size,max_sen_length,word_dim
        context_utterance_length_t = tf.transpose(context_utterance_length, perm=[1, 0])  # max_con_length, batch_size
        a = tf.split(context_t, hp.max_context_length, axis=0)  # 1,batch_size,max_sen_length,word_dim
        b = tf.split(context_utterance_length_t, hp.max_context_length, axis=0)  # 1,batch_size

        utterance_encodings = []
        for utterance, length in zip(a, b):
            utterance = tf.squeeze(utterance, axis=0)
            length = tf.squeeze(length, axis=0)
            _, rnn_final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, utterance,
                                                                         sequence_length=length,
                                                                         initial_state_fw=fw_cell.zero_state(
                                                                             batch_size, tf.float32),
                                                                         initial_state_bw=fw_cell.zero_state(
                                                                             batch_size, tf.float32))
            utterance_encoding = tf.concat(rnn_final_state, axis=1)
            utterance_encodings.append(tf.expand_dims(utterance_encoding, axis=0))

        utterance_encodings = tf.concat(utterance_encodings, axis=0)  # max_con_length,batch_size, 2*word_rnn_num_units

    with tf.variable_scope('context_encoder_layer',reuse=tf.AUTO_REUSE) as vs:
        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=random_seed+2)
        bias_initializer = tf.zeros_initializer()
        context_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.context_rnn_num_units,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)

        initialial_state = context_cell.zero_state(batch_size=batch_size,dtype=tf.float32)


        _,context_encoding = tf.nn.dynamic_rnn(cell=context_cell,inputs=utterance_encodings,sequence_length=context_length,initial_state=initialial_state,swap_memory=True,time_major=True)

    with tf.variable_scope('decoder_layer',reuse=tf.AUTO_REUSE) as vs:

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=random_seed+3)
        bias_initializer = tf.zeros_initializer()
        decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.decoder_rnn_num_units,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
        output_layer = layers_core.Dense(units=hp.vocab_size,activation=None,use_bias=False) #no activation and no bias

        if mode == modekeys.TRAIN:
            sequence_length = tf.constant(value=hp.max_sentence_length, dtype=tf.int32, shape=[batch_size])
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=response_in, sequence_length=sequence_length)
            decoder = BasicDecoder(cell=decoder_cell,helper=helper,initial_state=context_encoding,output_layer=output_layer)

            final_outputs, final_state, final_sequence_lengths = dynamic_decode(decoder=decoder,swap_memory=True,impute_finished=True)
            logit = final_outputs.rnn_output #[batch_size, max_res_size, vocab_size]
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=response_out,logits=logit)
            cross_entropy = tf.multiply(cross_entropy,response_mask)
            loss = tf.reduce_sum(cross_entropy,axis=1) # shouldn't divide num_steps
            loss = tf.reduce_mean(loss)
            l2_norm = hp.lambda_l2 * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])
            loss = loss + l2_norm

            debug_tensors = []
            return loss,debug_tensors

        elif mode == modekeys.EVAL:
            sequence_length = tf.constant(value=hp.max_sentence_length, dtype=tf.int32, shape=[batch_size])
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=response_in, sequence_length=sequence_length)
            decoder = BasicDecoder(cell=decoder_cell,helper=helper,initial_state=context_encoding, output_layer=output_layer)

            final_outputs, final_state, final_sequence_lengths = dynamic_decode(decoder=decoder,swap_memory=True)

            logit = final_outputs.rnn_output  # [batch_size, max_sen_size, vocab_size]
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=response_out, logits=logit)
            cross_entropy = tf.reduce_mean(tf.multiply(cross_entropy, response_mask))
            ppl = tf.exp(cross_entropy)
            return ppl

        elif mode == modekeys.PREDICT:
            if hp.beam_width == 0:
                helper = GreedyEmbeddingHelper(embedding=embedding_w,start_tokens=tf.constant(1, tf.int32, shape=[batch_size]), end_token=2)
                decoder = BasicDecoder(cell=decoder_cell, helper=helper, initial_state=context_encoding,output_layer=output_layer)
                final_outputs, final_state, final_sequence_lengths = dynamic_decode(decoder,
                                                                                    maximum_iterations=hp.max_sentence_length)
                results = {}
                results['response_ids'] = final_outputs.sample_id
                results['response_lens'] = final_sequence_lengths
                return results
            else:
                tiled_context_encoding = tile_batch(context_encoding, multiplier=hp.beam_width)
                decoder = BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=embedding_w,
                    start_tokens=tf.constant(value=1,dtype=tf.int32,shape=[batch_size]),
                    end_token=2,
                    initial_state=tiled_context_encoding,
                    beam_width=hp.beam_width,
                    output_layer=output_layer)


                final_outputs, final_state, final_sequence_lengths = dynamic_decode(decoder=decoder,swap_memory=True,maximum_iterations=hp.max_sentence_length)

                final_outputs = final_outputs.predicted_ids  # b,s,beam_width
                final_outputs = tf.transpose(final_outputs, perm=[0, 2, 1])  # b,beam_width,s
                # predicted_length = final_state.lengths #b,s
                predicted_length = None

                results = {}
                results['response_ids'] = final_outputs
                results['response_lens'] = None
                return results



def get_embedding_matrix(word_dim,mode,vocab_size,word_embed_path,vocab_path):
    if mode == modekeys.TRAIN:
        vocab, vocab_dict = helper.load_vocab(vocab_path)
        glove_vectors,glove_dict  = helper.load_glove_vectors(word_embed_path, vocab)
        initial_value = helper.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, word_dim,random_seed)
        embedding_w = tf.get_variable(name='embedding_W', initializer=initial_value, trainable=True)
    else:
        embedding_w = tf.get_variable(name='embedding_W',shape=[vocab_size,word_dim],dtype=tf.float32,trainable=True)
    return embedding_w