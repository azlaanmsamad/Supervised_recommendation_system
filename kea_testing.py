import pke
#from pke import compute_document_frequency
import glob
import os
#from strings import punctuations
import pandas as pd
import numpy as np
import pdb


def makeresultfolder():
    path = os.getcwd() + '/tagged_result'
    if os.path.exists(path):
        print('Result folder exists.')
    else:
        os.makedirs(path)
        print('Result folder created in the current directory.')

def makecsvfile():
    path = os.path.join(os.getcwd(), 'tagged_result', 'Tags.csv')
    with open(path, 'w+') as f:
        f.close()

def save_result(tags:list, textfilename: str):
    filename = os.getcwd() +  '/tagged_result/' + textfilename.split('.')[0] + '_tags' + '.csv'
    with open(filename, 'a') as f:
        wr = csv.writer(f)
        wr.writerows(tags)
        f.close()

def compute_document_frequency(input_file):
    #stopwords = list(punctuation)
    pke.compute_document_frequency(input_dir=input_file,
                           output_file='trial_doc_freq.tsv.gz',
                           extension='txt',           # input file extension
                           language='en',                # language of files
                           normalization="stemming")#,    # use porter stemmer
                           #stoplist=stoplist)

def test(files, number_of_tags, trained_model, test_DF_zip):
    # create a Kea extractor and set the input language to English (used for
    # the stoplist in the candidate selection method)
    extractor = pke.supervised.Kea()

    # load the content of the document, here in CoreNLP XML format
    # the use_lemmas parameter allows to choose using CoreNLP lemmas or stems 
    # computed using nltk
    extractor.load_document(files)
    
    # select the keyphrase candidates, for Kea the 1-3 grams that do not start or
    # end with a stopword.
    extractor.candidate_selection()
        
    # load the df counts
    df_counts = pke.load_document_frequency_file(input_file=test_DF_zip,
                                              delimiter='\t')
        
    # weight the candidates using Kea model.
    extractor.candidate_weighting(model_file=trained_model, df=df_counts)
        
    key_list = []
        
    for (keyphrase, score) in extractor.get_n_best(n=number_of_tags):
        key_list.append(keyphrase)

    files = files.split('/')[-1].split('.')[0]
    return files, key_list

    #save_dataframe(result_dict)
    #remove_doc_freq_file()

def result_initialiser():
    result_dict = {}
    result_dict['tags'] = []
    result_dict['filename'] = []
    return result_dict

def remove_doc_freq_file():
    os.remove('trial_doc_freq.tsv.gz')

def save_dataframe(results:dict):
    df = pd.DataFrame(results, columns=results.keys())
    path = os.getcwd() + '/tagged_result/' + 'Tags.csv'
    df.to_csv(path)

def tester(input_file_dir, number_of_tags):
    makeresultfolder()
    compute_document_frequency(input_file_dir)
    path = input_file_dir + '/*.txt'
    test_DF_zip = os.getcwd() + '/trial_doc_freq.tsv.gz'
    trained_model = 'kea_model.pickle'
    result_dict = result_initialiser()
    for files in glob.glob(path):
        filename, keyphrase = test(files, number_of_tags, trained_model, test_DF_zip)
        result_dict['tags'].append(keyphrase)
        result_dict['filename'].append(filename)
    save_dataframe(result_dict)


if __name__=='__main__':

    # Enter the name of the document here along with the extension, PLEASE ONLY USE .txt format files only.
    file_input_dir = ''

    path = os.getcwd() + '/' + str(file_input_dir)
    document_folder = path
    #Specify the number of tags you wish to generate for each document below.
    number_of_tags = 5
    #Evaluation
    print('Evaluation Started....')
    tester(document_folder, number_of_tags)
    print('Done.')

     
