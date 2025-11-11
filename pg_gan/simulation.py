"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import math
import msprime

# from stdpopsim
from .sps import engines
from .sps import species
from .sps import HomSap

# our imports
from . import global_vars

################################################################################
# SIMULATION
################################################################################

def im(params, sample_sizes, seed, reco):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # condense params
    N1 = params.N1.value
    N2 = params.N2.value
    T_split = params.T_split.value
    N_anc = params.N_anc.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = N1),
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = N2)]

    # no migration initially
    mig_time = T_split/2

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(
			time = T_split, source = 1, destination = 0, proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T_split, initial_size=N_anc,
            population_id=0)
	]

    # simulate tree sequence
    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def ooa2(params, sample_sizes,seed, reco):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # condense params
    T1 = params.T1.value
    T2 = params.T2.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = params.N3.value), # YRI is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = params.N2.value)] # CEU/CHB is second

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
        # change size of EUR
        msprime.PopulationParametersChange(time=T2,
            initial_size=params.N1.value, population_id=1),
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T1, source = 1, destination = 0,
            proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T1,
            initial_size=params.N_anc.value, population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def post_ooa(params, sample_sizes, seed, reco):
    """Note this is a 2 population model for CEU/CHB split"""
    assert len(sample_sizes) == 2

    # condense params
    T1 = params.T1.value
    T2 = params.T2.value
    mig = params.mig.value
    #m_EU_AS = params.m_EU_AS.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = params.N3.value), # CEU is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = params.N2.value)] # CHB is second

    # symmetric migration
    #migration_matrix=[[0, m_EU_AS],
    #                  [m_EU_AS, 0]]

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T2, source = 1, destination = 0,
            proportion = 1.0),
        # set mig rate to zero (need if using migration_matrix)
        #msprime.MigrationRateChange(time=T2, rate=0),
        # ancestral bottleneck
        msprime.PopulationParametersChange(time=T2,
            initial_size=params.N1.value, population_id=0),
        # ancestral size
        msprime.PopulationParametersChange(time=T1,
            initial_size=params.N_anc.value, population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
        #migration_matrix = migration_matrix,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def exp(params, sample_sizes, seed, reco):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0,
            growth_rate=params.growth.value),
        msprime.PopulationParametersChange(time=T2, initial_size=N2,
            growth_rate=0),
		msprime.PopulationParametersChange(time=params.T1.value,
            initial_size=params.N1.value)
	]

    ts = msprime.simulate(sample_size = sum(sample_sizes),
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def const(params, sample_sizes, seed, reco):
    assert len(sample_sizes) == 1

    # simulate data
    ts = msprime.simulate(sample_size=sum(sample_sizes), Ne=params.Ne.value,
        length=global_vars.L, mutation_rate=params.mut.value,
        recombination_rate=reco, random_seed = seed)

    return ts

def ooa3(params, sample_sizes, seed, reco):
    """From OOA3 as implemented in stdpopsim"""
    assert len(sample_sizes) == 3

    sp = species.get_species("HomSap")

    mult = global_vars.L/141213431 # chr9
    contig = sp.get_contig("chr9",length_multiplier=mult) # TODO vary the chrom

    # 14 params
    N_A = params.N_A.value
    N_B = params.N_B.value
    N_AF = params.N_AF.value
    N_EU0 = params.N_EU0.value
    N_AS0 = params.N_AS0.value
    r_EU = params.r_EU.value
    r_AS = params.r_AS.value
    T_AF = params.T_AF.value
    T_B = params.T_B.value
    T_EU_AS = params.T_EU_AS.value
    m_AF_B = params.m_AF_B .value
    m_AF_EU = params.m_AF_EU.value
    m_AF_AS = params.m_AF_AS.value
    m_EU_AS = params.m_EU_AS.value

    model = HomSap.ooa_3(N_A, N_B, N_AF, N_EU0, N_AS0, r_EU, r_AS, T_AF,
        T_B, T_EU_AS, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS)
    samples = model.get_samples(sample_sizes[0], sample_sizes[1],
        sample_sizes[2]) #['YRI', 'CEU', 'CHB']
    engine = engines.get_engine('msprime')
    ts = engine.simulate(model, contig, samples, seed=seed)

    return ts

def bottleneck(params, sample_sizes, seed, reco):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    N1 = params.N1.value # ancestral size
    N2 = params.N2.value # final size
    T1 = params.T1.value

    # should be negative if N1 is greater than N2
    g = -math.log(N1/N2)/T1
    #print("N1", N1, "N2", N2, "T1", T1, "growth", g)
    #input('enter')

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sum(sample_sizes),
        initial_size = params.N2.value)]

    demographic_events = [
        msprime.PopulationParametersChange(time=0,  initial_size=N2, growth_rate=g),
        msprime.PopulationParametersChange(time=T1, initial_size=N1, growth_rate=0),
	]

    ts = msprime.simulate(#sample_size = sum(sample_sizes),
        #population_size = params.N2.value,
        population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def dadi_joint_mig(params, sample_sizes, seed, reco):
    """Two population mosquito model from 2017 paper"""
    assert len(sample_sizes) == 2

    gen_per_year = 11

    # described past -> present
    NI = params.NI.value # the initial ancestral population size
    TG = params.TG.value*gen_per_year # the time of when the ancestral population begins to change in size
    NF = params.NF.value # the final ancestral population size, immediately prior to the split
    TS = params.TS.value*gen_per_year # the time of the split
    NI1 = params.NI1.value # the initial sizes of population 1 and population 2
    NI2 = params.NI2.value
    NF1 = params.NF1.value # the final sizes of these two populations
    NF2 = params.NF2.value
    MG = params.MG.value

    # compute growth rates from the start/end sizes and times
    # negative since backward in time
    g1 = -(1/TS) * math.log(NI1/NF1)
    g2 = -(1/TS) * math.log(NI2/NF2)
    g  = -(1/(TG-TS)) * math.log(NI/NF) # ancestral
    #small m
    MG = MG / 2 / NF

    demography = msprime.Demography()
    demography.add_population(name="POP1", initial_size=NF1)
    demography.add_population(name="POP2", initial_size=NF2)
    demography.add_population(name="ANC", initial_size=NF, initially_active=False)

    # dadi joint mig model
    demography.add_population_parameters_change(time=0, growth_rate=g1, population="POP1")
    demography.add_population_parameters_change(time=0, growth_rate=g2, population="POP2")

    demography.set_symmetric_migration_rate(populations=["POP1","POP2"], rate=MG)
    
    demography.add_population_split(time=TS, derived=["POP1", "POP2"], ancestral="ANC")
    demography.add_population_parameters_change(time=TS, growth_rate=g, population="ANC")
    demography.add_population_parameters_change(time=TG, growth_rate=0, population="ANC")

    #print(demography.debug())

    # If TG < TS, the model will throw an error even after sorting.
    if TS > TG:
        raise ValueError("Events must be time-sorted: TG must be greater than TS.", TS, TG)

    # simulate ancestry and mutations over that ancestry
    ts = msprime.sim_ancestry(
        samples = {'POP1':sample_sizes[0], 'POP2':sample_sizes[1]},
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=reco,
        ploidy=1,
        random_seed = seed) # keep it in haplotypes

    # TODO testing JC mutation model for multi-allelic sites
    mts = msprime.sim_mutations(ts, rate=params.mut.value, model="binary")
    # mts = msprime.sim_mutations(ts, rate=params.mut.value, model="JC69")

    return mts
