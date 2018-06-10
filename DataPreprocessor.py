import pandas as pd
import numpy as np
import time
import os
from string import digits
from collections import Counter
from itertools import chain

class DataPreprocessor(object):

    def __init__(self, csv_filename, max_cap_len, sep=' ',
                 word_freq_limit=2, save_path='data', feat_extractor='VGG16'):
        self.filename = csv_filename
        self.max_cap_len = max_cap_len
        self.separator = sep
        self.word_freq_limit = word_freq_limit
        self.save_path = save_path
        self.feat_extractor = feat_extractor

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

        self.log_file_size = None
        self.log_old_file_size = None
        self.log_difference = None

    def preprocess(self,filename):
        start_time = time.monotonic()
        self.load(filename=filename)



    def load(self,filename):
        file_path = os.path.join(self.DATA_FOLDER,self.MSCOCO_FOLDER,self.CLEANED,filename)
        df = pd.read_csv(file_path)
        df.sample(frac=1).reset_index(drop=True)
        self.image_files = df['file_path'].values
        self.captions = df['caption_1'].values
        self.remove_long_captions()
        self.get_word_frequencies()

    def remove_long_captions(self):
        print("Removing captions longer than", self.max_cap_len)
        reduced_images_files = []
        reduced_caption_files = []
        previous_file_size = len(self.captions)

        for i,caption in enumerate(self.captions):
            characterised_captions = self.clean_caption(caption=caption)
            if(len(characterised_captions)<= self.max_cap_len):
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


    def clean_caption(self,caption):
        incorrect_chars = digits + ";.,'/*?¿><:{}[\]|+"
        chars_ascii = str.maketrans('', '', incorrect_chars)
        quotes_ascii = str.maketrans('', '', '"')
        striped_caption = caption.strip().lower()
        striped_caption = striped_caption.translate(chars_ascii)
        striped_caption = striped_caption.translate(quotes_ascii)
        striped_caption = striped_caption.split(' ')
        return striped_caption

    def get_word_frequencies(self):
        self.word_frequencies = Counter(chain(*self.captions)).most_common()