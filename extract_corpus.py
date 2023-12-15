import re
import numpy as np
import pickle
import os

class FileData(object):
    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-16') as f:
          self.data = f.read()
          #print(self.data)

ABSOLUTE_PATH = r"E:\Python\language\Bi-directional-LSTM-Vietnamese-Spelling-AutoCorrection\corpus\Train_Full"

c_tri =  "/Chinh tri Xa hoi"

khoa_hoc = "/Khoa hoc"

kinh_doanh = "/Kinh doanh"

p_luat = "/Phap luat"

suc_khoe = "/Suc khoe"

the_thao = "/The thao"

corpus = [c_tri, khoa_hoc, kinh_doanh, p_luat, suc_khoe, the_thao]

#Extract folder path 
for folder_path in range(len(corpus)):
    corpus[folder_path] = ABSOLUTE_PATH + corpus[folder_path]

file_list = []

#Extracting text from corpus
for folder_path in corpus:
    count = 0
    for name in os.listdir(folder_path):
        count +=1
        if count == 1500:
          break        
        path = os.path.join(folder_path, name)
        if not os.path.isfile(path):
            continue
        file = FileData(path)
        file_list.append( file.data )

print('Corpus length: ', len(file_list))

#Save extracted corpus as Pickle file
path_corpus = r"E:\Python\language\Bi-directional-LSTM-Vietnamese-Spelling-AutoCorrection\corpus\train_corpus.pkl"

with open(path_corpus, 'wb') as pickle_file:
    pickle.dump(file_list, pickle_file)