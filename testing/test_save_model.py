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
pred_after = disc(new_pop, training=True)
print("pred_after", pred_after)
disc_optimizer = tf.keras.optimizers.Adam()

def discriminator_loss(real_output, fake_output):

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


# fake training
with tf.GradientTape() as disc_tape:
    # use current params
    real_regions = np.random.rand(1, 40, 36, 2)#, dtype=np.float32)
    generated_regions = np.random.rand(1, 40, 36, 2)#, dtype=np.float32)

    real_output = disc(real_regions, training=True)
    fake_output = disc(generated_regions, training=True)

    disc_loss = discriminator_loss(real_output, fake_output)

    # gradient descent
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
        disc.trainable_variables)
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator,
        disc.trainable_variables))