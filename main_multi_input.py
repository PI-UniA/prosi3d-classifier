#https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

import methods_multi_input_mixed_data as meth

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

from sklearn.model_selection import train_test_split
from keras.optimizers import adam_v2
from keras.layers import concatenate

import numpy as np
import locale

# print("[INFO] loading house attributes...")
df = meth.load_house_attributes('Houses Dataset/HousesInfo.txt')

# print("[INFO] loading house images...")
images = meth.load_house_images(df, 'Houses Dataset')
images = images / 255.0

# print("[INFO] processing data...")
(trainAttrX, testAttrX, trainImagesX, testImagesX) = train_test_split(df, images, test_size=0.25, random_state=42)

# find the largest house price in the training set and use it to scale our house prices to the range [0, 1]
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# house attributes data min-max scaling on continuous features, one-hot encoding on categorical features
trainAttrX, testAttrX = meth.process_house_attributes(df, trainAttrX, testAttrX)

# create the MLP and CNN models
mlp = meth.create_mlp(trainAttrX.shape[1], regress=False)
cnn = meth.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the *output* of both the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)


# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = adam_v2.Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(
	x=[trainAttrX, trainImagesX], y=trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])


# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))