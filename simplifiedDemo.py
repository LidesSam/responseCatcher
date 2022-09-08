#libreriad de redes neuronales
import tensorflow as tf
#para array numericos
import numpy as np

#ejemplo de red neuronal de regresion tomado de:
#https://www.youtube.com/watch?v=iX_on3VxZzk&t=633s
#red neuronal de regresion... toma entradas simples y devuleve un numero

#entradas de ejemplo
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
           "excelente entrega",
           "genial"
           "4",
           "entrega lenta",
           "entrega mal"]

tokenizer = Tokenizer(num_words=100) 
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 
print(word_index)

tokenzer = Tokenizer(num_words=100,"<unk>") #"<unk>" used to unknow words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index 

sequences= tokenizer.texts_to_sequences(sentences)
print(sequences)

#entradas de ejemplo
result=np.array([1,0,1,0,1,1,1,0,0],dtype=float)

#Dense inidica las conneciones entre cada neurana con las neuraonas de la capa siguiente
#solo una en este capa 
entrylayer= tf.keras.layers.Dense(units=1,input_shape=[1])
#determina el orden en el que se ejecuan las capas 
model =tf.keras.Sequential([entrylayer])


model.compile(
    #Adam es un algorimto
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss= "mean_squared_error"
    #" mean_square_error" 
    #una poca cantidad de erroresgrandes 
    # es peor que una gran catidad de errores grandes
    #
)

print("comenzando entrenamiento")

#fit indidca las vueltas
historial = model.fit(celsius,fahrenheit,epochs=500,verbose=False)
print("modelo Entrenado")




# para ver los resultados
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.xlabel("Magnitud perdida")
plt.plot(historial.history["loss"])
#

result= model.predict(["1"])
print("result:"+str(result))

result= model.predict(["mal])
print("result:"+str(result))

result= model.predict("excelente")
print("result:"+str(result))

result= model.predict(["bueno"])
print("result:"+str(result))