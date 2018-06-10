import numpy as np
import pickle
import h5py
import pandas as pd

class Generator(object):

    def __init__(self, data_path="data/", training_filename=None, validation_filename=None,
                 image_features_name=None, batch_size=100):
        self.data_path = data_path

        if training_filename == None:
            self.training_filename = self.data_path + 'training.txt'
        else:
            self.training_filename = training_filename

        if validation_filename == None:
            self.validation_filename = self.data_path + 'validation.txt'
        else:
            self.validation_filename = validation_filename

        if image_features_name == None:
            self.image_features_filename = self.data_path + 'VGG16_image_to_features.h5'
        else:
            self.image_features_filename = image_features_name

        self.dictionary = None
        self.training_dataset = None
        self.validation_dataset = None
        self.image_to_features = None

        self.logs = np.genfromtxt(self.data_path + 'data_parameters.log',delimiter=' ', dtype='str')

        self.logs = dict(zip(self.logs[:,0],self.logs[:,1]))

        self.MAX_TOKEN_LENGTH = int(self.logs['max_caption_length:'])+2
        self.IMG_FEATS = int(self.logs['IMG_FEATS:'])
        self.BOS = str(self.logs['BOS:'])
        self.PAD = str(self.logs['PAD:'])
        self.EOS = str(self.logs['EOS:'])
        self.VOCAB_SIZE = None
        self.wordtoix = None
        self.ixtoword = None
        self.BATCH_SIZE = batch_size

        self.load_dataset()
        self.load_vocabulary()
        self.load_image_features()

    def load_dataset(self):
        print("Loading saved train dataset...")
        train_data =  pd.read_table(self.training_filename,delimiter='*')
        train_data = np.asarray(train_data,dtype=str)
        self.training_dataset = train_data

        print("Loading saved validation dataset...")
        val_data = pd.read_table(self.validation_filename,delimiter='*')
        val_data = np.asarray(val_data,dtype=str)
        self.validation_dataset = val_data

    def load_vocabulary(self):
        print("Loading vocabulary... ")
        wordtoix = pickle.load(open(self.data_path + 'word_to_ix.p','rb'))
        ixtoword = pickle.load(open(self.data_path+'ix_to_word.p','rb'))
        self.VOCAB_SIZE = len(wordtoix)
        self.wordtoix = wordtoix
        self.ixtoword = ixtoword

    def load_image_features(self):
        self.image_to_features = h5py.File(self.image_features_filename,'r')

    def make_empty_batch(self):
        captions_batch = np.zeros((self.BATCH_SIZE,self.MAX_TOKEN_LENGTH,self.VOCAB_SIZE))
        images_batch = np.zeros((self.BATCH_SIZE,self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        target_batch = np.zeros((self.BATCH_SIZE, self.MAX_TOKEN_LENGTH,self.VOCAB_SIZE))
        return captions_batch,images_batch,target_batch

    def format_to_one_hot_enc(self,caption):
        tokenized_caption = caption.split()
        tokenized_caption = [self.BOS] + tokenized_caption + [self.EOS]
        one_hot_caption = np.zeros((self.MAX_TOKEN_LENGTH,self.VOCAB_SIZE))

        word_ids = [self.wordtoix[word] for word in tokenized_caption if word in self.wordtoix]

        for arg, word_id in enumerate(word_ids):
            one_hot_caption[arg,word_id] = 1
        return one_hot_caption

    def get_one_hot_target(self, one_hot_caption):
        one_hot_target = np.zeros_like(one_hot_caption)
        one_hot_target[:-1,:] = one_hot_caption[1:,:]
        return one_hot_target

    def get_image_features(self,image_name): #which is nothing but image_path from csv file
        image_features = self.image_to_features[image_name]['image_features'][:]
        image_input = np.zeros(self.MAX_TOKEN_LENGTH, self.VOCAB_SIZE)
        image_input[0,:] = image_features
        return image_input

    def wrap_in_dictionary(self,captions_batch,images_batch,targets_batch):
        return [{'text':captions_batch,'image':images_batch}, {'output':targets_batch}]

    def flow(self,mode):

        if(mode == 'train'):
            data = self.training_dataset
        if(mode == 'validation'):
            data = self.validation_dataset

        image_names = data[:,0].tolist()
        captions_batch,images_batch,targets_batch = self.make_empty_batch()
        counter = 0
        while True:
            for arg,image_name in enumerate(image_names):
                caption = data[:,1]
                one_hot_enc_caption = self.format_to_one_hot_enc(caption)
                captions_batch[counter,:,:] = one_hot_enc_caption
                targets_batch[counter,:,:] = self.get_one_hot_target(one_hot_caption=one_hot_enc_caption)
                images_batch[counter,:,:] = self.get_image_features(image_name)

                if counter == self.BATCH_SIZE -1:
                    yield_dictionary = self.wrap_in_dictionary(captions_batch,images_batch,targets_batch)
                    yield  yield_dictionary

                    captions_batch, images_batch, targets_batch = self.make_empty_batch()
                counter = counter + 1





