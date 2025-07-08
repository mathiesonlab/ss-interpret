import tf_keras as keras

for i in range(0,4):
    model = keras.models.load_model(f"models/CEU/CEU_{i}_230410")
    model.save_weights(f"models/CEU/CEU_{i}_230410_weights.h5")