# test keras save model

# python imports
import numpy as np
import tensorflow as tf

# our imports
import discriminator

SAVED_MODEL = "test_model.keras"
disc = discriminator.OnePopModel(10)
corrected = np.zeros((1, 20, 36, 2), dtype=np.float32)
pred_before = disc(corrected)
print("pred before", pred_before)
disc.save(SAVED_MODEL)

disc = tf.keras.models.load_model(SAVED_MODEL, custom_objects={"OnePopModel": discriminator.OnePopModel, "pop": 200}) # input_folder is a file in this case

corrected = np.zeros((1, 20, 36, 2), dtype=np.float32)
#corrected[0] = region
pred = disc(corrected, training=False)['output_1'].numpy()[0][0]
print("pred", pred)