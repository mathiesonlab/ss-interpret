# python imports
import tensorflow as tf
import time

# our imports
import param_set
from generator import Generator
import simulation
import global_vars
#import ss_helpers

disc_before = "/homes/smathieson/Documents/pg_gan_interpret/discriminators_og/CEU/CEU_0_230410"
disc_after = "/Users/smathieson/Dropbox/ss-interpret/trained_discs/CEU/CEU_0_230410_230830_finetuneAug23"

# simulate
print("sim exp")
exp_params = param_set.ParamSet(simulation.exp)
generator = Generator(simulation.exp, ["N1", "T1"], [198],
                        global_vars.DEFAULT_SEED)
generator.update_params([exp_params.N1.value, exp_params.T1.value])
mini_batch = generator.simulate_batch(1000, neg1=False)
print("x", mini_batch.shape)
#np.save("test_batch.npy", mini_batch)

# get predictions
disc = tf.saved_model.load(disc_before)
start = time.time()
preds = disc(mini_batch)
end = time.time()
print("time in ms", end-start)

# get stats
#mini_batch = np.load("test_batch.npy")
start = time.time()
stats = ss_helpers.stats_all(mini_batch)
end = time.time()
print("time in ms", end-start)



    