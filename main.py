# %%
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# %%
mnist = tf.keras.datasets.mnist

# %%
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()


# %%
xtrain = tf.keras.utils.normalize(xtrain, axis=1)
xtest = tf.keras.utils.normalize(xtest, axis=1)

# %%
xtrain = xtrain.reshape(-1, 28, 28, 1)
xtest = xtest.reshape(-1, 28, 28, 1)

# %%
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# %%
model.summary()

# %%
model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])


es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)

mc = ModelCheckpoint("./shaileshmodel.h5", monitor="val_acc",
                     verbose=1, save_best_only=True)

cb = [es, mc]


# %%
model.fit(xtrain, ytrain, epochs=5, validation_split=0.3, callbacks=cb)

# %%
test_loss, test_acc = model.evaluate(xtest, ytest)


# %%
print(test_loss)
print(test_acc)

# %%
prediction = model.predict([xtest])

# %%

# %%
img = cv2.imread('C:\\Users\\sherl\\Downloads\\num5.png')

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# %%
resize = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

# %%
plt.imshow(resize)

# %%
new_img = tf.keras.utils.normalize(resize, axis=1)

# %%
new_img = np.array(new_img).reshape(-1, 28, 28, 1)

# %%
new_img.shape

# %%
prediction = model.predict(new_img)

# %%
print(np.argmax(prediction))
