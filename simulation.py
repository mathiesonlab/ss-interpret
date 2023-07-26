"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import collections
import math
import msprime
import numpy as np

# from stdpopsim
#import sps.engines
#import sps.species
#import sps.HomSap

# our imports
import global_vars

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

def ooa2(params, sample_sizes,seed, reco): # also for fsc (fastsimcoal)
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

def three_epoch(params, sample_sizes, seed, reco):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    gen_per_year = 11

    Na = params.Na.value
    N1 = params.N1.value
    N2 = params.N2.value
    T1 = params.T1.value*gen_per_year
    T2 = params.T2.value*gen_per_year

    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N2),
        msprime.PopulationParametersChange(time=T2, initial_size=N1),
		msprime.PopulationParametersChange(time=T1, initial_size=Na)
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

'''
2-pop dadi model! (implement like baboon below)

The first model allowed for a phase of continuous exponential size change in the ancestral
population up until the time of the population split, after which each of the daughter
populations experienced their own exponential size change until the present. Migration
between daughter populations was not allowed.

The second model is identical to the first, except for the addition
of a symmetric, bidirectional migration parameter 2NIm, where NI is the initial ancestral
population size and m is the migration rate per gamete per generation. Both models also
include a parameter specifying the fraction of polymorphisms whose ancestral state was
misinferred (i.e. if the observed frequency is i out of n chromosomes, the true derived
allele frequency is n−i). In order to limit the number of free parameters, we fixed the
value of this parameter to 0.1%.
'''
def dadi_joint(params, sample_sizes, seed, reco):
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

    # compute growth rates from the start/end sizes and times
    # negative since backward in time
    g1 = -(1/TS) * math.log(NI1/NF1)
    g2 = -(1/TS) * math.log(NI2/NF2)
    g  = -(1/(TG-TS)) * math.log(NI/NF) # ancestral

    demography = msprime.Demography()
    demography.add_population(name="POP1", initial_size=NF1)
    demography.add_population(name="POP2", initial_size=NF2)
    demography.add_population(name="ANC", initial_size=NF, initially_active=False)

    # dadi joint model
    demography.add_population_parameters_change(time=0, growth_rate=g1, population="POP1")
    demography.add_population_parameters_change(time=0, growth_rate=g2, population="POP2")
    demography.add_population_split(time=TS, derived=["POP1", "POP2"], ancestral="ANC")
    demography.add_population_parameters_change(time=TS, growth_rate=g, population="ANC")
    demography.add_population_parameters_change(time=TG, growth_rate=0, population="ANC")

    #print(demography.debug())

    # simulate ancestry and mutations over that ancestry
    ts = msprime.sim_ancestry(
        samples = {'POP1':sample_sizes[0], 'POP2':sample_sizes[1]},
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=reco,
        ploidy=1) # keep it in haplotypes
    mts = msprime.sim_mutations(ts, rate=params.mut.value, model="binary")

    return mts

'''def ooa3(params, sample_sizes, seed, reco):
    """From OOA3 as implemented in stdpopsim"""
    assert len(sample_sizes) == 3

    sp = sps.species.get_species("HomSap")

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

    model = sps.HomSap.ooa_3(N_A, N_B, N_AF, N_EU0, N_AS0, r_EU, r_AS, T_AF,
        T_B, T_EU_AS, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS)
    samples = model.get_samples(sample_sizes[0], sample_sizes[1],
        sample_sizes[2]) #['YRI', 'CEU', 'CHB']
    engine = sps.engines.get_engine('msprime')
    ts = engine.simulate(model, contig, samples)

    return ts'''

def abc_dls_ooa(params, sample_sizes, seed, reco): # reco not used
    """
    This is the Model from Gravel et al. 2013 PNAS for Out of Africa

    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_B: Effective population size of Out of Africa (OOA) populations
        T_EU_AS: Time interval for separation of European and East Asian from now. in kilo year ago (kya)
        T_B: Time interval for separation between Africa and OOA populations from T_EU_AS. in kya
        T_AF: Time interval for decrease of effective population size of African population to ancestral effective
            population size from T_B. in kya
        m_AF_B: Bi-directional migration rate between African and OOA populations (x10^-5)
        m_AF_EU: Bi-directional migration rate between African and European populations (x10^-5)
        m_AF_AS: Bi-directional migration rate between African and East Asian populations (x10^-5)
        m_EU_AS: Bi-directional migration rate between European and East Asian populations (x10^-5)
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """
    #length=1e6
    mutation_rate=1.45e-8
    recombination_rate=1e-8

    # 14 params
    N_A = params.N_A.value
    N_B = params.N_B.value
    N_AF = params.N_AF.value
    N_EU0 = params.N_EU0.value
    N_AS0 = params.N_AS0.value
    T_AF = params.T_AF.value
    T_B = params.T_B.value
    T_EU_AS = params.T_EU_AS.value
    m_AF_B = params.m_AF_B .value
    m_AF_EU = params.m_AF_EU.value
    m_AF_AS = params.m_AF_AS.value
    m_EU_AS = params.m_EU_AS.value

    # we used rates before
    N_EU = params.N_EU.value
    N_AS = params.N_AS.value
    #r_EU = params.r_EU.value
    #r_AS = params.r_AS.value

    '''# testing
    param_names = "N_A N_B N_AF N_EU0 N_AS0 T_AF T_B T_EU_AS m_AF_B m_AF_EU " +
        "m_AF_AS m_EU_AS N_EU N_AS"
    param_names = param_names.split()
    param_values = [N_A, N_B, N_AF, N_EU0, N_AS0, T_AF, T_B, T_EU_AS, m_AF_B,
        m_AF_EU, m_AF_AS, m_EU_AS, N_EU, N_AS]
    assert len(param_names) == len(param_values)
    to_print = [param_names[i] + ": " + str(param_values[i]) for i in range(len(param_names))]
    for p in to_print:
        print(p)
    input('enter')'''

    (n1, n2, n3) = sample_sizes

    T_EU_AS, T_B, T_AF = np.array([T_EU_AS, T_B, T_AF]) * (1e3 / 29.0)
    m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS = np.array([m_AF_B, m_AF_EU, m_AF_AS,
        m_EU_AS]) * 1e-5
    r_EU = (math.log(N_EU / N_EU0) / T_EU_AS)
    r_AS = (math.log(N_AS / N_AS0) / T_EU_AS)
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS)
    ]
    migration_matrix = [
        [0, m_AF_EU, m_AF_AS],
        [m_AF_EU, 0, m_EU_AS],
        [m_AF_AS, m_EU_AS, 0],
    ]
    demographic_events = [
        # CEU and CHB merge into B with rate changes at T_EU_AS
        msprime.MassMigration(
            time=T_EU_AS, source=2, destination=1, proportion=1.0),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
        msprime.PopulationParametersChange(
            time=T_EU_AS, growth_rate=0, population_id=2),
        # Population B merges into YRI at T_B
        msprime.MassMigration(
            time=T_B + T_EU_AS, source=1, destination=0, proportion=1.0),
        msprime.MigrationRateChange(time=T_B + T_EU_AS, rate=0),
        # Size changes to N_A at T_AF
        msprime.PopulationParametersChange(
            time=T_AF + T_B + T_EU_AS, initial_size=N_A, population_id=0)]
    ts = msprime.simulate(
        population_configurations=population_configurations,
        migration_matrix=migration_matrix,
        demographic_events=demographic_events, length=global_vars.L,
            mutation_rate=mutation_rate,
        recombination_rate=recombination_rate)
    return ts

def bndx(params, sample_sizes, seed, reco): # reco not used
    """
    This is the back to Africa model with Neanderthal to OOA population, Denisova or Unknown to East Asia and
        African archaic to African populations.

    :param params: all the parameters necessary for this model in a list or array format
        N_A: The ancestral effective population size
        N_AF: Modern effective population size of Africa population
        N_EU: Modern effective population size of European population
        N_AS: Modern effective population size of East Asian population
        N_EU0: Effective population size of European population before exponential growth.
        N_AS0: Effective population size of East Asian population before exponential growth.
        N_BC: Effective population size of Back to Africa migrated population.
        N_B: Effective population size of Out of Africa (OOA) populations
        N_AF0: Effective population size of African populations before Back to Africa migration
        T_DIntro: Time interval for introgression in East Asian from Denisova or Unknown from now. in kilo year ago
            (kya)
        T_EU_AS: Time interval for separation of European and East Asian from T_DIntro. in kya
        T_NIntro: Time interval for introgression in OOA from Neanderthal from T_EU_AS. in kya
        T_XIntro: Time interval for introgression in African population from African archaic from T_EU_AS. in kya
        T_Mix: Time interval for mixing with Back to Africa population from T_EU_AS. in kya
        T_Sep: Time interval for separation of Back to Africa population from OOA from T_Mix. in kya
        T_B: Time interval for separation between Africa and OOA populations from maximum between T_NIntro, T_XIntro and
            T_Sep. in kya
        T_AF: Time interval for decrease of effective population size of African population to ancestral effective
            population size from T_B. in kya
        T_N_D: Time interval for separation between Neanderthal and Denisova or Unknwon from now. in kya
        T_H_A: Time interval for separation between Neanderthal and modern humans from T_N_D. in kya
        T_H_X: Time interval for separation between African archaic and modern humans from now. in kya
        mix: the fraction of African genome is replaced but Back to Africa population
        nintro: the fraction of introgression happened to OOA populations.
        dintro: the fraction of introgression happened to East Asians
        xintro: the fraction of introgression happened to African populations
    :param inds: the number of haplotypes per populations for example (10,10,10)
    :param length: the length of chromosome that has to be simulated. default is 1mb region
    :param mutation_rate: the amount of mutation rate. default is 1.45x10^-8 per generation per base
    :param recombination_rate: the amount of recombination rate. default is 10^-8 per generation per base
    :param replicates: the number of replicated of length chromosome. default is 300
    :return: will return the msprime simulations. which then can be used to extract SFS
    """

    #length=1e6
    mutation_rate=1.45e-8
    recombination_rate=1e-8

    # 24 params
    N_A = params.N_A.value
    N_AF = params.N_AF.value
    N_EU = params.N_EU.value
    N_AS = params.N_AS.value
    N_EU0 = params.N_EU0.value
    N_AS0 = params.N_AS0.value
    N_BC = params.N_BC.value
    N_B = params.N_B.value
    N_AF0 = params.N_AF0.value
    T_DIntro = params.T_DM.value # T_DIntro == T_DM
    T_EU_AS = params.T_EU_AS.value
    T_NIntro = params.T_NM.value # T_NIntro == T_NM
    T_XIntro = params.T_XM.value # T_XIntro == T_XM
    T_Mix = params.T_Mix.value
    T_Sep = params.T_Sep.value
    T_B = params.T_B.value
    T_AF = params.T_AF.value
    T_N_D = params.T_N_D.value
    T_H_A = params.T_H_A.value
    T_H_X = params.T_H_X.value
    mix = params.Mix.value/100     # convert from percent to fraction
    nintro = params.NMix.value/100 # convert from percent to fraction
    dintro = params.DMix.value/100 # convert from percent to fraction
    xintro = params.XMix.value/100 # convert from percent to fraction

    (n1, n2, n3) = sample_sizes
    T_DIntro, T_EU_AS, T_NIntro, T_XIntro, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X = np.array(
        [T_DIntro, T_EU_AS, T_NIntro, T_XIntro, T_Mix, T_Sep, T_B, T_AF, T_N_D, T_H_A, T_H_X]) * (1e3 / 29.0)

    events = {}
    AFR, EUR, ASN, GST, NEA, DEN, XAF = 0, 1, 2, 3, 4, 5, 6

    r_EU = (math.log(N_EU / N_EU0) / (T_DIntro + T_EU_AS))
    r_AS = (math.log(N_AS / N_AS0) / (T_DIntro + T_EU_AS))
    # N_B=N_EU0+N_AS0
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=n1, initial_size=N_AF),
        msprime.PopulationConfiguration(
            sample_size=n2, initial_size=N_EU, growth_rate=r_EU),
        msprime.PopulationConfiguration(
            sample_size=n3, initial_size=N_AS, growth_rate=r_AS),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_BC),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_A)
    ]

    # Denisova or unknown admixture
    events['deni_intro_asn'] = T_DIntro
    deni_intro_asn = [msprime.MassMigration(
        time=events['deni_intro_asn'], source=ASN, destination=DEN, proportion=dintro)]
    # CEU and CHB merge into B with rate changes at T_EU_AS
    events['split_eu_as'] = events['deni_intro_asn'] + T_EU_AS
    split_eu_as = [msprime.MassMigration(
        time=events['split_eu_as'], source=ASN, destination=EUR, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['split_eu_as'], initial_size=N_B, growth_rate=0, population_id=EUR),
        msprime.MigrationRateChange(time=events['split_eu_as'], rate=0)]
    # introgression
    events['nean_intro_eur'] = T_NIntro + events['split_eu_as']
    nean_intro_eur = [msprime.MassMigration(
        time=events['nean_intro_eur'], source=EUR, destination=NEA, proportion=nintro)]
    # introgression XAFR
    events['xafr_intro_afr'] = T_XIntro + events['split_eu_as']
    xafr_intro_afr = [msprime.MassMigration(
        time=events['xafr_intro_afr'], source=AFR, destination=XAF, proportion=xintro)]

    # back migration
    events['back_migration'] = events['split_eu_as'] + T_Mix
    back_migration = [msprime.MassMigration(time=events['back_migration'], source=AFR,
                                            destination=GST, proportion=mix), msprime.PopulationParametersChange(
        time=events['back_migration'], initial_size=N_AF0, population_id=AFR)]
    # spearation between back and OOA
    events['split_ooa_back'] = events['back_migration'] + T_Sep
    split_ooa_back = [msprime.MassMigration(time=events['split_ooa_back'], source=GST,
                                            destination=EUR, proportion=1.0)]
    # Population B merges into YRI at T_B
    events['split_afr_ooa'] = max(events['split_ooa_back'], events['xafr_intro_afr'],
                                  events['nean_intro_eur']) + T_B
    split_afr_ooa = [msprime.MassMigration(
        time=events['split_afr_ooa'], source=EUR, destination=AFR, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_afr_ooa'], rate=0)]
    # Size changes to N_A at T_AF
    events['ancestral_size'] = events['split_afr_ooa'] + T_AF
    ancestral_size = [msprime.PopulationParametersChange(
        time=events['ancestral_size'], initial_size=N_A, population_id=AFR)]
    # Denisova or Unknwon merging with Neanderthal
    events['neanderthal_denisova'] = T_N_D
    neanderthal_denisova = [msprime.MassMigration(
        time=events['neanderthal_denisova'], source=DEN, destination=NEA, proportion=1.0),
        msprime.PopulationParametersChange(
            time=events['neanderthal_denisova'], initial_size=N_A, population_id=NEA)]
    # Neanderthal merging with humans
    events['human_neanderthal'] = T_N_D + T_H_A
    human_neanderthal = [msprime.MassMigration(
        time=events['human_neanderthal'], source=NEA, destination=AFR, proportion=1.0)]
    # XAFR merging with humans
    events['human_xafr'] = T_H_X
    human_xafr = [msprime.MassMigration(
        time=events['human_xafr'], source=XAF, destination=AFR, proportion=1.0)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)

    '''dd = msprime.DemographyDebugger(
	       	population_configurations=population_configurations,
        	#migration_matrix=migration_matrix,
        	demographic_events=demographic_events)
    dd.print_history()
    input('enter')'''

    ts = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events, length=global_vars.L, mutation_rate=mutation_rate,
        recombination_rate=recombination_rate)

    return ts

'''
def baboon_old(params, sample_sizes, seed, reco):
    # TODO redo w/o mass migrations and with population_split and admixture
    """
    Baboon demography from Figure 4A https://www.science.org/doi/10.1126/sciadv.aau6947
    Note: sample sizes (in individuals) for real data should be:
    P. anubis       ANU 3 (use 3 since one admixed)
    P. cynocephalus CYN 2
    P. hamadryas    HAM 2
    P. kindae       KIN 3
    P. papio        PAP 2
    P. ursinus      URS 2
    T. gelada       GEL 1 # outgroup, not currently implemented in model below
    """

    # constants
    mutation_rate = 0.9e-8 # per base pair per generation
    recombination_rate = 1e-8 # per base pair per generation (TODO made this up)
    gen_time = 11.0 # years
    Ne = 100 # constant across all pops for now TODO

    # in KYA
    # using "Tb" for "baboon" to distinguish from human times
    T_A = params.Tb_A.value # interval between events F and A in the figure
    #T_B = params.Tb_B.value # time of ghost split
    T_C = params.Tb_C.value # interval between events D and C in the figure
    T_D = params.Tb_D.value # interval between events G and D in the figure
    T_E = params.Tb_E.value # interval between events G and E in the figure
    T_F = params.Tb_F.value # interval between events J and F in the figure
    T_G = params.Tb_G.value # time of KIN forming from admixture
    #T_H = params.Tb_H.value # time of additional admixture
    # no T_I apparently
    T_J = params.Tb_J.value # ANU and PAP split
    #T_K = params.Tb_K.value # PAP admixture

    # convert times to generations
    T_A, T_C, T_D, T_E, T_F, T_G, T_J = np.array(
        [T_A, T_C, T_D, T_E, T_F, T_G, T_J]) / gen_time # not in ky

    # set pop sizes the same for now
    N_ANU = N_CYN = N_HAM = N_KIN = N_PAP = N_URS = Ne

    # KINS and KINN are the ghost populations that come together to create KIN
    ANU, CYN, HAM, KIN, PAP, URS, KINS, KINN = 0, 1, 2, 3, 4, 5, 6, 7
    #ANU, CYN, HAM, PAP, URS = 0, 1, 2, 3, 4 # needs to match pop config order below
    (n1, n2, n3, n4, n5, n6) = sample_sizes

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=n1, initial_size=N_ANU),
        msprime.PopulationConfiguration(sample_size=n2, initial_size=N_CYN),
        msprime.PopulationConfiguration(sample_size=n3, initial_size=N_HAM),
        msprime.PopulationConfiguration(sample_size=n4, initial_size=N_KIN),
        msprime.PopulationConfiguration(sample_size=n5, initial_size=N_PAP),
        msprime.PopulationConfiguration(sample_size=n6, initial_size=N_URS),
        msprime.PopulationConfiguration(sample_size=0, initial_size=0),
        msprime.PopulationConfiguration(sample_size=0, initial_size=0)]

    events = {}

    # ANU and PAP merge (event J)
    events['split_anu_pap'] = T_J
    split_anu_pap = [
        msprime.MassMigration(time=events['split_anu_pap'], source=ANU,
            destination=PAP, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_anu_pap'], rate=0),
        msprime.PopulationParametersChange(time=events['split_anu_pap'],
            initial_size=Ne, growth_rate=0, population_id=PAP)]

    # KIN admixture (event G)
    events['kin_admix'] = T_G
    kin_admix = [
        msprime.MassMigration(time=events['kin_admix'], source=KIN, # south
            destination=KINS, proportion=0.52),
        msprime.MassMigration(time=events['kin_admix'], source=KIN, # north
            destination=KINN, proportion=0.48),
        msprime.MigrationRateChange(time=events['kin_admix'], rate=0),
        msprime.PopulationParametersChange(time=events['kin_admix'],
            initial_size=0, growth_rate=0, population_id=KIN),
        msprime.PopulationParametersChange(time=events['kin_admix'],
            initial_size=Ne, growth_rate=0, population_id=KINS),
        msprime.PopulationParametersChange(time=events['kin_admix'],
            initial_size=Ne, growth_rate=0, population_id=KINN)]

    # URS and KIN-south-ghost merge (event D)
    events['split_urs_kinS'] = events['kin_admix'] + T_D
    split_urs_kinS = [
        msprime.MassMigration(time=events['split_urs_kinS'], source=KINS,
            destination=URS, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_urs_kinS'], rate=0),
        msprime.PopulationParametersChange(time=events['split_urs_kinS'],
            initial_size=Ne, growth_rate=0, population_id=URS)]

    # CYN and URS+KIN-south-ghost merge (event C)
    events['split_cyn_urs'] = events['split_urs_kinS'] + T_C
    split_cyn_urs = [
        msprime.MassMigration(time=events['split_cyn_urs'], source=CYN,
            destination=URS, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_cyn_urs'], rate=0),
        msprime.PopulationParametersChange(time=events['split_cyn_urs'],
            initial_size=Ne, growth_rate=0, population_id=URS)]

    # ANU and PAP merge into HAM (event F)
    events['northern_split'] = events['split_anu_pap'] + T_F
    northern_split = [
        msprime.MassMigration(time=events['northern_split'], source=PAP,
            destination=HAM, proportion=1.0),
        msprime.MigrationRateChange(time=events['northern_split'], rate=0),
        msprime.PopulationParametersChange(time=events['northern_split'],
            initial_size=Ne, growth_rate=0, population_id=HAM)]

    # KIN-north-ghost and HAM merge (event E)
    events['split_kinN_ham'] = events['kin_admix'] + T_E
    split_kinN_ham = [
        msprime.MassMigration(time=events['split_kinN_ham'], source=KINN,
            destination=HAM, proportion=1.0),
        msprime.MigrationRateChange(time=events['split_kinN_ham'], rate=0),
        msprime.PopulationParametersChange(time=events['split_kinN_ham'],
            initial_size=Ne, growth_rate=0, population_id=HAM)]

    # all join, i.e. URS merge into HAM (event A)
    events['all_split'] = events['northern_split'] + T_A
    all_split = [
        msprime.MassMigration(time=events['all_split'], source=URS,
            destination=HAM, proportion=1.0),
        msprime.MigrationRateChange(time=events['all_split'], rate=0),
        msprime.PopulationParametersChange(time=events['all_split'],
            initial_size=Ne, growth_rate=0, population_id=HAM)]

    demographic_events = []
    for event in collections.OrderedDict(sorted(events.items(), key=lambda x: x[1])).keys():
        demographic_events = demographic_events + eval(event)

    dd = msprime.DemographyDebugger(
	       	population_configurations=population_configurations,
        	demographic_events=demographic_events)
    dd.print_history()
    input('enter')

    ts = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events,
        length=global_vars.L,
        mutation_rate=mutation_rate,
        recombination_rate=0)#recombination_rate)

    return ts'''

def baboon(params, sample_sizes, seed, reco):
    """
    Baboon demography from Figure 4A https://www.science.org/doi/10.1126/sciadv.aau6947
    Note: sample sizes (in individuals) for real data should be:
    P. anubis       ANU 3 (use 3 since one admixed)
    P. cynocephalus CYN 2
    P. hamadryas    HAM 2
    P. kindae       KIN 3
    P. papio        PAP 2
    P. ursinus      URS 2
    T. gelada       GEL 1 # outgroup, not currently implemented in model below
    """

    # constants
    gen_time = 11.0 # years
    Ne = 100 # constant across all pops for now TODO

    # in years
    # using "Tb" for "baboon" to distinguish from human times
    T_A = params.Tb_A.value # interval between events F and A in the figure
    #T_B = params.Tb_B.value # time of ghost split
    T_C = params.Tb_C.value # interval between events D and C in the figure
    T_D = params.Tb_D.value # interval between events G and D in the figure
    #T_E = params.Tb_E.value # interval between events G and E in the figure
    T_F = params.Tb_F.value # interval between events J and F in the figure
    T_G = params.Tb_G.value # time of KIN forming from admixture
    #T_H = params.Tb_H.value # time of additional admixture
    # no T_I apparently
    T_J = params.Tb_J.value # ANU and PAP split
    #T_K = params.Tb_K.value # PAP admixture

    # convert times to generations
    T_A, T_C, T_D, T_F, T_G, T_J = np.array(
        [T_A, T_C, T_D, T_F, T_G, T_J]) / gen_time # not in ky

    # set pop sizes the same for now
    N_ANU = N_CYN = N_HAM = N_KIN = N_PAP = N_URS = Ne

    # new way with msprime.Demography()
    demography = msprime.Demography()
    demography.add_population(name="ANU", initial_size=N_ANU)
    demography.add_population(name="CYN", initial_size=N_CYN)
    demography.add_population(name="HAM", initial_size=N_HAM)
    demography.add_population(name="KIN", initial_size=N_KIN)
    demography.add_population(name="PAP", initial_size=N_PAP)
    demography.add_population(name="URS", initial_size=N_URS)
    demography.add_population(name="KINS", initial_size=Ne, initially_active=False) # KIN south
    demography.add_population(name="KINN", initial_size=Ne, initially_active=False) # KIN north
    demography.add_population(name="AP", initial_size=Ne, initially_active=False) # ANU-PAP
    demography.add_population(name="UK", initial_size=Ne, initially_active=False) # URS-KINS
    demography.add_population(name="S", initial_size=Ne, initially_active=False) # south ancestor
    demography.add_population(name="N", initial_size=Ne, initially_active=False) # north ancestor
    demography.add_population(name="ANC", initial_size=Ne, initially_active=False) # MRCA

    # KIN admixture (event G)
    demography.add_admixture(time=T_G, derived="KIN", ancestral=["KINS", "KINN"], proportions=[0.52, 0.48])

    # ANU and PAP merge (event J)
    demography.add_population_split(time=T_J, derived=["ANU", "PAP"], ancestral="AP")

    # URS and KIN-south-ghost merge (event D)
    split_urs_kinS = T_G + T_D
    demography.add_population_split(time=split_urs_kinS, derived=["URS", "KINS"], ancestral="UK")

    # CYN and URS+KIN-south-ghost merge (event C)
    split_cyn_urs = split_urs_kinS + T_C
    demography.add_population_split(time=split_cyn_urs, derived=["CYN", "UK"], ancestral="S")

    # KIN-north-ghost and HAM merge (event E)
    #split_kinN_ham = T_G + T_E
    #demography.add_population_split(time=split_kinN_ham, derived=["KINN", "HAM"], ancestral="N")

    # TODO E and F are the same right now!!
    # ANU and PAP merge into HAM (event F)
    northern_split = T_J + T_F
    demography.add_population_split(time=northern_split, derived=["KINN", "HAM", "AP"], ancestral="N")

    # all join, i.e. URS merge into HAM (event A)
    all_split = northern_split + T_A
    demography.add_population_split(time=all_split, derived=["S", "N"], ancestral="ANC")

    # in case any are out of order
    demography.sort_events()

    #print(demography.debug())

    # simulate ancestry and mutations over that ancestry
    # sample_sizes are in haplotypes - convert to individuals for msprime
    for n in sample_sizes:
        assert n % 2 == 0
    indv_sizes = [n//2 for n in sample_sizes]
    (n1, n2, n3, n4, n5, n6) = indv_sizes
    ts = msprime.sim_ancestry(
        samples = {'ANU':n1, 'CYN':n2, 'HAM':n3, 'KIN':n4, 'PAP':n5, 'URS':n6},
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=reco)
    mts = msprime.sim_mutations(ts, rate=params.mut.value)

    return mts

'''def admix():

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=100)
    demography.add_population(name="B", initial_size=100)
    demography.add_population(name="ADMIX", initial_size=100)
    demography.add_population(name="ANC", initial_size=100)
    demography.add_admixture(
        time=10, derived="ADMIX", ancestral=["A", "B"], proportions=[0.25, 0.75])
    demography.add_population_split(time=20, derived=["A", "B"], ancestral="ANC")
    print(demography.debug())'''
