import pke
#import logging
## Training the model on train set.
#train_input_dir = 'drive/My Drive/Recommendation systems/kea_trained/train_doc/'
reference_file = 'drive/My Drive/Recommendation systems/kea_trained/reference.txt'
output_mdl = "drive/My Drive/Recommendation systems/kea_trained/Models/kea_model.pickle"
#train_df_file = 'drive/My Drive/Recommendation systems/kea_trained/train_DF.tsv.gz'

#logging.info('Loading df counts from {}'.format(df_file))

df_counts = pke.load_document_frequency_file(input_file='train_DF.tsv.gz',
                                             delimiter='\t')

pke.train_supervised_model(input_dir='train_doc/',
                           reference_file='reference.txt',
                           model_file='model/kea_model.pickle',
                           extension='txt',
                           language='en',
                           normalization="stemming",
                           df=df_counts,
                           model=pke.supervised.Kea())
