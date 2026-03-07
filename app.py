import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)

loss, acc = model.evaluate(x_test, y_test, verbose=0)

st.title(" Classificação de Dígitos com TensorFlow + Streamlit")
st.write(f"Acurácia do modelo: **{acc:.2f}**")

st.write("Insira um dígito (0 a 9) para ver previsão em uma imagem aleatória:")

digit = st.number_input("Número desejado", min_value=0, max_value=9, step=1)

if st.button("Gerar previsão"):

    idx = np.where(np.argmax(y_test, axis=1) == digit)[0]
    random_idx = np.random.choice(idx)
    image = x_test[random_idx]

    st.image(image, caption=f"Dígito real: {digit}", width=150)

    pred = model.predict(image.reshape(1, 28, 28))
    st.write(f"Previsão do modelo: **{np.argmax(pred)}**")
