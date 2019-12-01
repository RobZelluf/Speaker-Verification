from utils import *
from keras_model import *
from sklearn.preprocessing import OneHotEncoder

## Hyperparameters
epochs = 20
batch_size = 128

X, Y = read_data()
m = X.shape[0]
num_speakers = max(Y + 1)
print("Num speakers:", num_speakers)

X = np.swapaxes(X, 1, 2)

onehot_encoder = OneHotEncoder(sparse=False)
Y = Y.reshape(-1, 1)
Y = onehot_encoder.fit_transform(Y)

model = GRU_NN(num_speakers)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X, Y, batch_size=batch_size, epochs=epochs)

print(history)