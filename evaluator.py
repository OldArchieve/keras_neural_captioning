import os
import numpy as np
import pandas as pd
import pickle
import h5py


class Evaluator(object):

    def __init__(self, model, type_path='validation',
                 data_path=os.path.join("data", "processed"),
                 images_path=os.path.join("data", "mscoco", "val2017"),
                 log_filename="data_parameters.log",
                 test_data_filename="test_data.txt",
                 word_to_ix='word_to_ix.p',
                 ix_to_word='ix_to_word.p',
                 image_to_feats_filename='VGG16_image_to_features.h5'):
        self.model = model
        self.data_path = data_path
        self.images_path = images_path
        self.log_filename = log_filename

        if type_path == 'validation':
            self.TYPE_PATH = 'val'
        else:
            self.TYPE_PATH = 'train'

        logs = self._load_log_file()
        self.BOS = str(logs['BOS:'])
        self.EOS = str(logs['EOS:'])
        self.IMG_FEATS = int(logs['IMG_FEATS:'])
        self.MAX_TOKEN_LENGTH = int(logs['max_caption_length:']) + 2
        self.test_data = pd.read_table(os.path.join(data_path, self.TYPE_PATH, test_data_filename), sep="*")
        self.word_to_ix = pickle.load(open(os.path.join(data_path, self.TYPE_PATH, word_to_ix), 'rb'))
        self.ix_to_word = pickle.load(open(os.path.join(data_path, self.TYPE_PATH, ix_to_word), 'rb'))
        self.VOCAB_SIZE = int(len(self.ix_to_word))
        self.images_to_feats = h5py.File(os.path.join(data_path, self.TYPE_PATH,image_to_feats_filename))

    def _load_log_file(self):
        logs = np.genfromtxt(os.path.join(self.data_path, self.TYPE_PATH, self.log_filename), delimiter=' ',
                             dtype='str')
        logs = dict(zip(logs[:, 0], logs[:, 1]))
        return logs

    def display_captions(self, image_file=None, data_name=None):

        test_data = self.test_data

        if image_file == None:
            image_name = np.asarray(test_data.sample(1))[0][0]
        else:
            image_name = image_file

        features = self.images_to_feats[image_name]['image_features'][:]
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCAB_SIZE))
        bos_token_id = self.word_to_ix[self.BOS]
        text[0, 0, bos_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        print(self.BOS)

        for arg in range(self.MAX_TOKEN_LENGTH - 1):
            predictions = self.model.predict([text, image_features])

            word_id = np.argmax(predictions[0, arg, :])
            #print(word_id)
            next_word_arg = arg + 1
            text[0, next_word_arg, arg] = 1
            #print(text.shape)
            word = self.ix_to_word[word_id]
            print(word)
            if word == self.EOS:
                break

        return image_name
