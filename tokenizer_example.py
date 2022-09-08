import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


#https://www.youtube.com/watch?v=fNxaJsNG3-s

#extrae las palabras de un arreglo de sentencias y las indexa en un arreglo numerioc
# esto es nesesario para poder procesar texto en un ared neuronal


#https://www.youtube.com/watch?v=r9QjkdSJZ2g
# secuanciar la fraces


sentences=["buen servicio",
           "1",
           "entrega rapida",
           "3",
           "mal servicio",
           "exelente entrega",
           "genial"
           "4",
           "entrega lenta",]
tokenizer = Tokenizer(num_words=100) 
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 
print(word_index)

tokenzer = Tokenizer(num_words=100) 
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 

sequences= tokenizer.texts_to_sequences(sentences)
print(sequences)