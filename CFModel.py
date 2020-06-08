# A simple implementation of matrix factorization for collaborative filtering expressed as a Keras Sequential model

# Keras uses TensorFlow tensor library as the backend system to do the heavy compiling

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Reshape, dot, Input
from tensorflow.keras.models import Sequential, Model

user_input = Input(shape=(1,))
user_embedding = Embedding(max_userid, K_FACTORS, input_length=1)(user_input)
user_embedding = Reshape((K_FACTORS,))(user_embedding)

# Q is the embedding layer that creates a Movie by latent factors matrix.
# If the input is a movie_id, Q returns the latent factor vector for that movie.
movie_input = Input(shape=(1,))
movie_embedding = Embedding(max_movieid, K_FACTORS, input_length=1)(movie_input)
movie_embedding = Reshape((K_FACTORS,))(movie_embedding)

dot_product = dot([user_embedding, movie_embedding], axes=1)

model = Model(inputs=[user_input, movie_input], outputs=dot_product)
model.compile(loss='mse', optimizer='adamax')

# The rate function to predict user's rating of unrated items
def rate(self, user_id, item_id):
    return self.predict([np.array([user_id]), np.array([item_id])])[0][0]