from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense
from keras.layers import Add, LSTM
from keras.regularizers import l2

class MyModel(object):

    def __init__(self, max_token_len=16, vocab_size=1024):
        print("Initiating the model")
        self.max_token_len = max_token_len
        self.vocab_size = vocab_size

    def captioner(self,rnn='lstm', no_img_feats=4096,hidden_size=512, emb_size=512, regularizer=1e-8):
        max_token_len = self.max_token_len
        vocab_size = self.vocab_size
        text_input = Input(shape=(max_token_len, vocab_size), name='text')
        text_mask = Masking(mask_value=0.0, name='text_mask')(text_input)
        text_to_emb = TimeDistributed(Dense(units=emb_size, kernel_regularizer=l2(regularizer),
                                            name='text_emb'))(text_mask)

        text_dropout = Dropout(0.5, name='text_dropout')(text_to_emb)

        image_input = Input(shape=(max_token_len, no_img_feats), name="image")
        image_emb = TimeDistributed(Dense(units=emb_size, kernel_regularizer=l2(regularizer),
                                          name='image_emb'))(image_input)

        image_dropout = Dropout(0.5, name='image_dropout')(image_emb)

        recurrent_inputs = [text_dropout, image_dropout]

        merged_inputs = Add()(recurrent_inputs)

        if rnn == 'lstm':
            recurrent_network = LSTM(units=hidden_size,
                                     recurrent_regularizer=l2(regularizer),
                                     bias_regularizer=l2(regularizer),
                                     kernel_regularizer=l2(regularizer),
                                     return_sequences=True,
                                     name="recurrent_network")(merged_inputs)

        output = TimeDistributed(Dense(units=vocab_size,
                                       kernel_regularizer=l2(regularizer),
                                       activation='softmax',
                                       name='output'))(recurrent_network)

        inputs = [text_input, image_input]

        model = Model(inputs=inputs, outputs=output)
        return model

