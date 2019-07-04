# *_*coding:utf-8 *_*

import tensorflow as tf
import numpy as np
import librosa
import os
import re
from model import WaveNet
from utils import SpeechLoader


def test_preprocess():
    test_path = '/home/ydf_micro/datasets/data_thchs30/test'

    def handle_file(dirpath, filename):
        if filename.endswith('.wav') or filename.endswith('.WAV'):
            filename_path = os.path.join(dirpath, filename)
            label_path = filename_path + '.trn'
            if os.stat(filename_path).st_size < 240000:
                return None, None

            return filename_path, label_path
        else:
            return None, None

    def label_text(path):
        path = re.sub(r'test', 'data', path)
        name = os.path.basename(path).split('.')[0]
        with open(path, 'r', encoding='utf-8') as f:
            # 文件中一个有三行,第一行为文字内容(text),
            # 第二行为音节(syllable),第三行为音素(phoneme)
            text = f.readline()  # get text

            return name, text.strip('\n')

    wav_files = []
    labels_dict = {}
    for dirpath, dirnames, filenames in os.walk(test_path):
        for filename in filenames:
            file, label = handle_file(dirpath, filename)
            if file:
                wav_files.append(file)
                id, text = label_text(label)
                labels_dict[id] = text

    print('测试初始样本数: ', len(wav_files))  # 样本数
    print('测试标签长度: ', len(labels_dict))

    return wav_files, labels_dict


def speech_to_text(wav_files, labels_dict):
    n_mfcc = 60

    # load data

    speech_loader = SpeechLoader(n_mfcc=n_mfcc, is_training=False)

    wav_max_len = 673

    # load model
    model = WaveNet(speech_loader.vocab_size, n_mfcc=n_mfcc, is_training=False)

    saver = tf.train.Saver(tf.trainable_variables())

    test_wav = wav_files[:10]


    # word dict
    word_map = {value: key for key, value in speech_loader.wordmap.items()}
    print(word_map)

    with tf.Session() as sess:

        saver.restore(sess, tf.train.latest_checkpoint('../model'))

        for wav_path in test_wav:
            wav, sr = librosa.load(wav_path, mono=True)
            mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr, n_mfcc=n_mfcc),
                                               axis=0), [0, 2, 1])
            mfcc = mfcc.tolist()

            while len(mfcc[0]) < wav_max_len:
                mfcc[0].append([0] * n_mfcc)

            # recognition
            decoded = tf.transpose(model.logit, perm=[1, 0, 2])
            decoded, probs = tf.nn.ctc_beam_search_decoder(decoded, model.seq_len,
                                                           top_paths=1, merge_repeated=True)
            predict = tf.sparse_to_dense(decoded[0].indices,
                                         decoded[0].dense_shape,
                                         decoded[0].values) + 1
            output, probs = sess.run([predict, probs], feed_dict={model.input_data: mfcc})

            # result
            words = ''
            for i in range(len(output[0])):
                words += word_map.get(output[0][i], -1)

            wav_name = os.path.basename(wav_path).split('.')[0]

            print('-------------------------------------------------------')
            print(f'Input: {wav_path}')
            print(f'Output: {words}')
            print(f'True result: {labels_dict[wav_name]}')


if __name__ == '__main__':
    wav_files, labels_dict = test_preprocess()
    speech_to_text(wav_files, labels_dict)