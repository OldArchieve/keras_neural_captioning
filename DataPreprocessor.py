import pandas as pd
import numpy as np
import time
import os
from string import digits
from collections import Counter
from itertools import chain
import h5py
import pickle


class DataPreprocessor(object):

    def __init__(self, image_size, csv_filename, max_cap_len, sep=' ',
                 word_freq_limit=2, save_path='data/', feat_extractor='VGG16', extract_feats=False):
        self.filename = csv_filename
        self.max_cap_len = max_cap_len
        self.separator = sep
        self.word_freq_limit = word_freq_limit
        self.save_path = save_path
        self.feat_extractor = feat_extractor
        self.extract_feats = extract_feats

        self.word_frequencies = None
        self.BOS = '<S>'
        self.EOS = '<E>'
        self.PAD = '<P>'

        self.wordtoix = None
        self.ixtoword = None
        self.elasped_time = None
        self.DATA_FOLDER = 'data'
        self.MSCOCO_FOLDER = 'mscoco'
        self.CLEANED = 'cleaned'

        self.image_files = None
        self.captions = None
        self.feats = None
        self.extracted_features = None

        self.log_file_size = None
        self.log_old_file_size = None
        self.log_difference = None
        self.log_current_vocab_size = None
        self.log_initial_vocab_size = None
        self.log_removed_vocab_size = None
        self.image_size = image_size
        self.elapsed_time = None

    def load(self, filename):
        file_path = os.path.join(self.DATA_FOLDER, self.MSCOCO_FOLDER, self.CLEANED, filename)
        df = pd.read_csv(file_path)
        df.sample(frac=1).reset_index(drop=True)
        self.image_files = df['file_path'].values
        self.captions = df['caption_1'].values

    def remove_long_captions(self):
        print("Removing captions longer than", self.max_cap_len)
        reduced_images_files = []
        reduced_caption_files = []
        previous_file_size = len(self.captions)

        for i, caption in enumerate(self.captions):
            characterised_captions = self.clean_caption(caption=caption)
            if (len(characterised_captions) <= self.max_cap_len):
                reduced_caption_files.append(self.captions[i])
                reduced_images_files.append(self.image_files[i])
        self.captions = reduced_caption_files
        self.image_files = reduced_images_files

        new_file_size = len(self.captions)
        size_difference = previous_file_size - new_file_size
        print("Removed files :: ", size_difference)
        print("New file size :: ", new_file_size)
        print("Old file size :: ", previous_file_size)
        self.log_difference = size_difference
        self.log_file_size = new_file_size
        self.log_old_file_size = previous_file_size

    def clean_caption(self, caption):
        incorrect_chars = digits + ";.,'/*?¿><:{}[\]|+"
        chars_ascii = str.maketrans('', '', incorrect_chars)
        quotes_ascii = str.maketrans('', '', '"')
        striped_caption = caption.strip().lower()
        striped_caption = striped_caption.translate(chars_ascii)
        striped_caption = striped_caption.translate(quotes_ascii)
        striped_caption = striped_caption.split(' ')
        return striped_caption

    # this is same as calculating word to index
    # captions = ["hello", "word", "hello"]
    # returns--> [('hello', 2), ('word', 1)]
    def get_word_frequencies(self):
        self.word_frequencies = Counter(chain(*self.captions)).most_common()
        self.word_frequencies = np.asarray(self.word_frequencies)

    def construct_dictionaries(self):
        words = self.word_frequencies[:, 0]
        self.wordtoix = {self.PAD: 0, self.BOS: 1, self.EOS: 2}
        self.wordtoix.update({word: word_id for word_id, word in enumerate(words, 3)})
        self.ixtoword = {word_id: word for word, word_id in self.wordtoix.items()}

    def remove_infrequent_words(self):
        if self.word_freq_limit != 0:
            print("Removing words which are less than the word limit", self.word_freq_limit)
            for i, word_list in enumerate(self.word_frequencies):
                word_frequency = word_list[1]

                if word_frequency <= self.word_freq_limit:
                    final_arg = i
                    break  # here the assumption is that the self.word_frequencies is always in the descending order
        else:
            self.word_frequencies = self.word_frequencies
        previous_word_freq_len = self.word_frequencies.shape[0]  # all rows
        self.word_frequencies = self.word_frequencies[0:final_arg]
        new_word_freq_len = self.word_frequencies.shape[0]

        difference = previous_word_freq_len - new_word_freq_len

        self.log_current_vocab_size = new_word_freq_len
        self.log_initial_vocab_size = previous_word_freq_len
        self.log_removed_vocab_size = difference

    def extract_image_features(self):
        from keras.preprocessing import image
        from keras.models import Model
        if self.feat_extractor == 'VGG16':
            from keras.applications.vgg16 import preprocess_input
            from keras.applications import VGG16
            self.IMG_FEATS = 4096
            base_model = VGG16(weights='imagenet')
            model = Model(input=base_model.input,
                          output=base_model.get_layer('fc2').output)
            self.extracted_features = []
            number_of_images = len(self.image_files)
            for arg, image_path in enumerate(self.image_files):
                if arg % 100 == 0:
                    print("Extracted {0} percentage of images".format((arg / number_of_images) * 100))
                img = image.load_img(image_path, target_size=(self.image_size, self.image_size))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)

    def save_feats_as_hd5(self):
        print("Saving extracted features into hd5 file")
        data_file = h5py.File(self.save_path + self.feat_extractor + 'image_to_features.h5')
        number_of_images = len(self.image_files)

        for arg, image_path in enumerate(self.image_files):
            file_id = data_file.create_group(image_path)
            image_data = file_id.create_dataset('image_features', (self.IMG_FEATS,), dtype='float32')
            image_data[:] = self.extracted_features[arg, :]

            if arg % 100 == 0:
                print("Number of images saved", arg)
                print("Number of images remaining", number_of_images - arg)
        data_file.close()

    def save_captions(self):
        data_file = open(self.save_path + "caption.txt", "w")
        data_file.write("image_path*caption\n")

        for arg, image_path in enumerate(self.image_files):
            caption = ' '.join(self.captions[arg])
            data_file.write('{0}*{1}'.format(image_path, caption))
        data_file.close()

    def write_dictionaries(self):
        pickle.dump(self.wordtoix, open(self.save_path + 'word_to_ix.p', 'wb'))
        pickle.dump(self.ixtoword, open(self.save_path + 'ix_to_word.p', 'wb'))

    def preprocess(self, filename):
        start_time = time.monotonic()
        self.load(filename=filename)
        self.remove_long_captions()
        self.get_word_frequencies()
        self.remove_infrequent_words()
        self.construct_dictionaries()
        if self.extract_feats == True:
            self.extract_image_features()
            self.save_feats_as_hd5()

        self.save_captions()
        self.write_dictionaries()
        self.elapsed_time = time.monotonic() - start_time
        print("Elasped Time is ", self.elapsed_time)