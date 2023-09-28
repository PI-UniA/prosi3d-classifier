from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

from keras.layers import concatenate


# Input A: 32 - 8 - 4
# Input B: 128 - 64 - 32 - 4

# define two sets of inputs
inputA = Input(shape=(32,)) 
inputB = Input(shape=(128,))

# # the first branch operates on the first input
x = Dense(8, activation="relu")(inputA)
x = Dense(4, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# # the second branch opreates on the second input
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(4, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# # combine the output of the two branches
combined = concatenate([x.output, y.output]) #4-dim + 4-dim --> 8-dim

# # apply a FC layer and then a regression prediction on the
# # combined outputs
z = Dense(2, activation="relu")(combined) # 8 --> 2
z = Dense(1, activation="linear")(z) # 2 --> 1

# # our model will accept the inputs of the two branches and
# # then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)
