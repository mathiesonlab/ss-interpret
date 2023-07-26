"""
Generator class for pg-gan.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from numpy.random import default_rng
import traceback

# our imports
#import abc_dls_best
import global_vars
import param_set
import simulation
import util

################################################################################
# GENERATOR CLASS
################################################################################

class Generator:

    def __init__(self, simulator, param_names, sample_sizes, seed,
        mirror_real=False, reco_folder=""):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_samples = sum(sample_sizes) # should match haps *or* indvs
        self.rng = default_rng(seed)
        self.curr_params = None

        self.pretraining = False

        # for real data, use HapMap
        if mirror_real and reco_folder != None:
            files = global_vars.get_reco_files(reco_folder)
            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)

        else:
            self.prior, self.weights = [], []

    def simulate_batch(self, batch_size=global_vars.BATCH_SIZE, params=[],
        region_len=False, real=False, neg1=True, ball=False, param_objs=None):
        # if ball==True, sample a ball around current parameters
        # NOTE: if ball==True, params should be empty (use current), and param_objs
        #       should be specified

        # initialize 4D matrix (two channels for distances)
        if region_len:
            regions = []
        else:
            regions = np.zeros((batch_size, self.num_samples,
                global_vars.NUM_SNPS, 2), dtype=np.float32) # two channels

        # set up parameters
        sim_params = param_set.ParamSet(self.simulator)
        if real:
            pass # keep orig for "fake" real
        elif params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, params)

        # simulate each region
        for i in range(batch_size):
            seed = self.rng.integers(1,high=2**32) # like GAN "noise"

            # sample params around a ball
            if ball:
                assert params == [] # use current params
                s_proposal = []
                for k in range(len(self.curr_params)):
                    s_proposal.append(param_objs[k].proposal(self.curr_params[k], 0.05)) # small T
                sim_params.update(self.param_names, s_proposal)

            ts = self.simulator(sim_params, self.sample_sizes, seed,
                self.get_reco(sim_params))
            region = prep_region(ts, neg1, region_len=region_len)

            if region_len:
                regions.append(region)
            else:
                regions[i] = region

        return regions

    def real_batch(self, batch_size = global_vars.BATCH_SIZE, neg1=True,
        region_len=False):
        return self.simulate_batch(batch_size=batch_size, real=True, neg1=neg1,
            region_len=region_len)

    def update_params(self, new_params):
        self.curr_params = new_params

    def get_reco(self, params):
        if self.prior == []:
            return params.reco.value

        return draw_background_rate_from_prior(self.prior, self.weights)

def draw_background_rate_from_prior(prior_rates, prob):
    return np.random.choice(prior_rates, p=prob)

def prep_region(ts, neg1, region_len):
    """Gets simulated data ready"""
    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L for j in
        range(snps_total-1)]

    # when mirroring real data
    return util.process_gt_dist(gt_matrix, dist_vec, region_len=region_len,
        neg1=neg1)

# testing
if __name__ == "__main__":

    '''print("sim exp")
    exp_params = param_set.ParamSet()
    generator = Generator(simulate_exp, ["N1", "T1"], [20],
                          global_vars.DEFAULT_SEED)
    generator.update_params([exp_params.N1.value, exp_params.T1.value])
    mini_batch = generator.simulate_batch()
    print("x", mini_batch.shape)

    # test ABC-DLS (OOA)
    print("sim OOA")
    ooa_params = abc_dls_best.OOA
    generator = Generator(simulation.abc_dls_ooa, list(ooa_params.keys()),
        [20,20,20], global_vars.DEFAULT_SEED)
    generator.update_params(list(ooa_params.values()))
    mini_batch = generator.simulate_batch()
    print("x", mini_batch.shape)

    # test ABC-DLS (BNDX)
    print("sim BNDX")
    bndx_params = abc_dls_best.BNDX
    generator = Generator(simulation.bndx, list(bndx_params.keys()), [20,20,20],
        global_vars.DEFAULT_SEED)
    generator.update_params(list(bndx_params.values()))
    mini_batch = generator.simulate_batch()
    print("x", mini_batch.shape)

    # print params
    bndx_pset = param_set.ParamSet(simulation.bndx)
    bndx_pset.update(generator.param_names, generator.curr_params)
    bndx_pset.event_times_bndx()


    # test baboon
    print("sim baboon")
    sample_sizes = [3,2,2,3,2,2] # individuals here since divided already
    print(sample_sizes)

    # set up parameters and generato
    baboon_param_dict = abc_dls_best.BABOON
    param_str = ",".join(list(baboon_param_dict.keys()))
    baboon_params = util.parse_params(param_str, simulation.baboon)
    generator = Generator(simulation.baboon, param_str.split(","), sample_sizes,
                          global_vars.DEFAULT_SEED, phased=False)

    # try random parameters
    num_batch = 500
    for i in range(num_batch):
        if i % 10 == 0:
            print(i, "/", num_batch)
        s_random = [param.start() for param in baboon_params]
        generator.update_params(s_random)

        # simulate batch
        try:
            mini_batch = generator.simulate_batch()
        except:
            baboon_pset = param_set.ParamSet(simulation.baboon)
            baboon_pset.update(generator.param_names, generator.curr_params)
            baboon_pset.event_times_baboon()
        #print("x", mini_batch.shape)

    # test unphased
    ts = simulation.const(param_set.ParamSet(simulation.const), [10],
        global_vars.DEFAULT_SEED, 1e-8)
    region = prep_region(ts, False, False, phased=False)
    print(region.shape)'''

    # test dadi_joint
    sample_sizes = [12, 10]
    ts = simulation.dadi_joint(param_set.ParamSet(simulation.dadi_joint),
        sample_sizes, global_vars.DEFAULT_SEED, 8.4e-09)
    region = prep_region(ts, False, False)
    print(region.shape)
