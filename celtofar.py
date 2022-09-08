#libreriad de redes neuronales
import tensorflow as tf
#para array numericos
import numpy as np

#ejemplo de red neuronal de regresion tomado de:
#https://www.youtube.com/watch?v=iX_on3VxZzk&t=633s
#red neuronal de regresion... toma entradas simples y devuleve un numero

#entradas de ejemplo
celsius=np.array([-40,-20,100,15,285,-26.6667123,231,124,-15],dtype=float)
#entradas de ejemplo
fahrenheit=np.array([-40,-4,212,59,545,-16,447.8,255.2,5],dtype=float)

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

result= model.predict(["aloha"])
print("result:"+str(result))