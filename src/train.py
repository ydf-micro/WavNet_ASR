# *_*coding:utf-8 *_*

from utils import SpeechLoader
from model import WaveNet
import tensorflow as tf
import time
import os


def train():
    '''

    :return:
    '''

    batch_size = 8
    n_mfcc = 60
    n_epoch = 100

    source_file = '/home/ydf_micro/datasets/data_thchs30'
    speech_loader = SpeechLoader(os.path.join(source_file, 'train'),
                                 batch_size, n_mfcc)

    n_out = speech_loader.vocab_size


    # load model

    model = WaveNet(n_out, batch_size=batch_size, n_mfcc=n_mfcc)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.graph.finalize() # Graph is read-only after this statement

        for epoch in range(n_epoch):
            speech_loader.create_batches()  # random shuffle data
            speech_loader.reset_batch_pointer()
            for batch in range(speech_loader.n_batches):
                batch_start = time.time()
                batches_wav, batches_label = speech_loader.next_batch()
                feed = {model.input_data: batches_wav, model.targets: batches_label}
                train_loss, _ = sess.run([model.cost, model.optimizer_op], feed_dict=feed)
                batch_end = time.time()
                print(f'epoch: {epoch+1}/{n_epoch}, batch: {batch+1}/{speech_loader.n_batches}, '
                      f'loss: {train_loss:.2f}, time: {(batch_end-batch_start):.2f}s')

            # save models
            if epoch % 5 == 0:
                saver.save(sess, os.path.join(os.path.dirname(os.getcwd()),
                                              'model', 'speech.module'), global_step=epoch)


if __name__ == '__main__':

    start = time.time()

    train()

    end = time.time()
    print(f'训练时间:{(end - start) / 60:.2f}分钟')