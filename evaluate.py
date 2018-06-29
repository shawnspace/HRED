import tensorflow as tf
import input_layer
import modekeys
import hparam
import HRED
from tensorflow.python.training import saver as saver_lib
import bleu
import numpy as np
from tensorflow.core.framework import summary_pb2
import os



def evaluate(eval_file,model_dir,summary_dir,train_steps):
    hp = hparam.create_hparam()

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        input_features = input_layer.create_input_layer(filename=eval_file,hp=hp,mode=modekeys.EVAL)

        ppl  = HRED.impl(input_features=input_features,hp=hp,mode=modekeys.EVAL)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        saver = tf.train.Saver()
        checkpoint = saver_lib.latest_checkpoint(model_dir)
        saver.restore(sess=sess,save_path=checkpoint)
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        tf.logging.info(msg='Begin evaluation')

        try:
            total_ppl = 0
            eval_step = 0
            while not coord.should_stop():
                perplexity = sess.run(fetches=ppl)
                total_ppl += perplexity
                eval_step += 1
        except tf.errors.OutOfRangeError:
            avg_ppl = total_ppl / eval_step
            tf.logging.info('Finish evaluation. The perplexity is {}'.format(avg_ppl))
            write_to_summary(summary_dir, 'eval_ppl', avg_ppl, train_steps)
        finally:
            coord.request_stop()
        coord.join(threads)


        return avg_ppl

def calculate_bleu_score(generate_response, reference_response): # should remove padding elements
    #reference_corpus is like [[[token1, token2, token3]]]
    reference_corpus = [[ref.tolist()] for ref in reference_response]
    #translation corpus is like [[token1, token2]]
    translation_corpus = [gen.tolist() for gen in generate_response]
    result = bleu.compute_bleu(reference_corpus=reference_corpus,translation_corpus=translation_corpus)
    return result[0]

def write_to_summary(output_dir,summary_tag,summary_value,current_global_step):
    summary_writer = tf.summary.FileWriterCache.get(output_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = summary_tag
    if isinstance(summary_value, np.float32) or isinstance(summary_value, float):
        value.simple_value = float(summary_value)
    elif isinstance(summary_value,int) or isinstance(summary_value, np.int64) or isinstance(summary_value, np.int32):
        value.simple_value = int(summary_value)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()





