# test keras save model

# python imports
import numpy as np
import tensorflow as tf

# our imports
import discriminator

SAVED_MODEL = "test_model.keras"
disc = discriminator.OnePopModel()#20)
corrected = np.random.rand(1, 20, 36, 2)#, dtype=np.float32)
print(corrected.shape)
pred_before = disc(corrected)
print("pred before", pred_before)
disc.save(SAVED_MODEL)

disc = tf.keras.models.load_model(SAVED_MODEL, custom_objects={"OnePopModel": discriminator.OnePopModel}) # input_folder is a file in this case

#corrected = np.zeros((1, 20, 36, 2), dtype=np.float32)
new_pop = np.random.rand(1, 40, 36, 2)#, dtype=np.float32)
#corrected[0] = region
pred_after = disc(new_pop, training=False)
print("pred_after", pred_after)