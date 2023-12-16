from collections import Counter
from keras.models import load_model
from nltk import ngrams,word_tokenize
import numpy as np
import re
import string

model = load_model('E:\Python\language\Bi-directional-LSTM-Vietnamese-Spelling-AutoCorrection\spelling.h5')

NGRAM=2
MAXLEN=40 
alphabet = ['\x00', ' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'Đ']
letters=list("abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđABCDEFGHIJKLMNOPQRSTUVWXYZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÉÈẺẼẸÊẾỀỂỄỆÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴĐ")
accepted_char=list((string.digits + ''.join(letters)))

def call(sentence):
    def encoder_data(text, maxlen=MAXLEN):
            text = "\x00" + text
            x = np.zeros((maxlen, len(alphabet)))
            for i, c in enumerate(text[:maxlen]):
                x[i, alphabet.index(c)] = 1
            if i < maxlen - 1:
              for j in range(i+1, maxlen):
                x[j, 0] = 1
            return x    
            
    def decoder_data(x):
        x = x.argmax(axis=-1)
        return ''.join(alphabet[i] for i in x)

    def nltk_ngrams(words, n=2):
        return ngrams(words.split(), n)

    def guess(ngram):
        text = ' '.join(ngram)
        preds = model.predict(np.array([encoder_data(text)]), verbose=0)
        return decoder_data(preds[0]).strip('\x00')

    def correct(sentence):
        for i in sentence:
            if i not in accepted_char:
                sentence=sentence.replace(i," ")
        ngrams = list(nltk_ngrams(sentence, n=NGRAM))
        guessed_ngrams = list(guess(ngram) for ngram in ngrams)

        print("N gram", ngrams)
        print("guess", guessed_ngrams)

        candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
        for nid, ngram in (enumerate(guessed_ngrams)):
            for wid, word in (enumerate(re.split(' +', ngram))):
                candidates[nid + wid].update([word])
   
        output = ' '.join(c.most_common(1)[0][0] for c in candidates)  
        return output

    guess = correct(sentence)

    return guess

#sentence = "Hom lay tooi đi hoc"
sentence ='Tôi ko yêu thích họk ngôn ngữ mớj'
#sentence = 'Thnh pố Hồ Chí Mình là mộp địa điểm du lịch nổi tiếg'
guess = call(sentence)
print(guess)