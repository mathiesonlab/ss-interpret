import sys
import tf_keras as keras

for i in range(int(sys.argv[1]), int(sys.argv[2])):
    model = keras.models.load_model(f"models/CEU/CEU_{i}_230410")
    model.save_weights(f"models/CEU/CEU_{i}_230410.h5")