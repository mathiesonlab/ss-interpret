"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from scipy.stats import norm
import sys

# our imports
import simulation

class Parameter:
    """
    Holds information about evolutionary parameters to infer.
    Note: the value arg is NOT the starting value, just used as a default if
    that parameter is not inferred, or the truth when training data is simulated
    """

    def __init__(self, value, min, max, name):
        self.value = value
        self.min = min
        self.max = max
        self.name = name
        self.proposal_width = (self.max - self.min)/15 # heuristic

    def __str__(self):
        s = '\t'.join(["NAME", "VALUE", "MIN", "MAX"]) + '\n'
        s += '\t'.join([str(self.name), str(self.value), str(self.min),
            str(self.max)])
        return s

    def start(self):
        # random initialization
        return np.random.uniform(self.min, self.max)

    def start_range(self):
        start_min = np.random.uniform(self.min, self.max)
        start_max = np.random.uniform(self.min, self.max)
        if start_min <= start_max:
            return [start_min, start_max]
        return self.start_range()

    def fit_to_range(self, value):
        value = min(value, self.max)
        return max(value, self.min)

    def proposal(self, curr_value, multiplier):
        if multiplier <= 0: # last iter
            return curr_value

        # normal around current value (make sure we don't go outside bounds)
        new_value = norm(curr_value, self.proposal_width*multiplier).rvs()
        new_value = self.fit_to_range(new_value)
        # if the parameter hits the min or max it tends to get stuck
        if new_value == curr_value or new_value == self.min or new_value == \
            self.max:
            return self.proposal(curr_value, multiplier) # recurse
        else:
            return new_value

    def proposal_range(self, curr_lst, multiplier):
        new_min = self.fit_to_range(norm(curr_lst[0], self.proposal_width *
            multiplier).rvs())
        new_max = self.fit_to_range(norm(curr_lst[1], self.proposal_width *
            multiplier).rvs())
        if new_min <= new_max:
            return [new_min, new_max]
        return self.proposal_range(curr_lst, multiplier) # try again

class ParamSet:

    def __init__(self, simulator):
        """Takes in a simulator to determine which params are needed"""

        # const (mosquito right now)
        if simulator == simulation.const:
            # using 500k - 3M based on estimate of roughly 1M in 2017 paper
            self.Ne = Parameter(1e6, 500e3, 3e6, "Ne")
            self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        # exp (human right now -> change N1/N2/reco/mut for mosquito)
        elif simulator == simulation.exp:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            #self.N1 = Parameter(1e6, 500e3, 3e6, "N1")
            #self.N2 = Parameter(1e6, 500e3, 3e6, "N2")
            self.T1 = Parameter(2000, 1500, 5000, "T1")
            self.T2 = Parameter(350, 100, 1500, "T2")
            self.growth = Parameter(0.005, 0.0, 0.05, "growth")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

            #self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            #self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        # three_epoch (mosquito right now)
        elif simulator == simulation.three_epoch:
            d = 1000
            self.Na = Parameter(384845.04236326, 384845.04236326-d, 384845.04236326+d, "Na")
            self.N1 = Parameter(1891371.2275129908, 1891371.2275129908-d, 1891371.2275129908+d, "N1")
            self.N2 = Parameter(11140821.633397933, 11140821.633397933-d, 11140821.633397933+d, "N2")
            self.T1 = Parameter(60447.09280337712, 60447.09280337712-d, 60447.09280337712+d, "T1")
            self.T2 = Parameter(22708.95299729848, 22708.95299729848-d, 22708.95299729848+d, "T2")

            self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        # mosquito dadi joint models: GNS_vs_BFS (1st line)
        elif simulator == simulation.dadi_joint:
            d = 1000
            self.NI = Parameter(420646, 420646-d, 420646+d, "NI")
            self.TG = Parameter(89506, 89506-d, 89506+d, "TG")
            self.NF = Parameter(9440437, 9440437-d, 9440437+d, "NF")
            self.TS = Parameter(2245, 2245-d, 2245+d, "TS")
            self.NI1 = Parameter(18328570, 18328570-d, 18328570+d, "NI1")
            self.NI2 = Parameter(42062652, 42062652-d, 42062652+d, "NI2")
            self.NF1 = Parameter(42064645, 42064645-d, 42064645+d, "NF1")
            self.NF2 = Parameter(42064198, 42064198-d, 42064198+d, "NF2")

            self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        # im
        elif simulator == simulation.im:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
            self.T_split = Parameter(2000, 500, 20000, "T_split")
            self.mig = Parameter(0.05, -0.2, 0.2, "mig")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # ooa2
        elif simulator == simulation.ooa2:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N3 = Parameter(12000, 1000, 30000, "N3")
            self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
            self.T1 = Parameter(2000, 1500, 5000, "T1")
            self.T2 = Parameter(350, 100, 1500, "T2")
            self.mig = Parameter(0.05, -0.2, 0.2, "mig")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # postOOA
        elif simulator == simulation.postOOA:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N3 = Parameter(12000, 1000, 30000, "N3")
            self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
            self.T1 = Parameter(2000, 1500, 5000, "T1")
            self.T2 = Parameter(350, 100, 1500, "T2")
            self.mig = Parameter(0.05, -0.2, 0.2, "mig")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # ooa3 (dadi)
        elif simulator == simulation.ooa3:
            self.N_A = Parameter(None, 1000, 30000, "N_A")
            self.N_B = Parameter(2100, 1000, 20000, "N_B")
            self.N_AF = Parameter(12300, 1000, 40000, "N_AF")
            self.N_EU0 = Parameter(1000, 100, 20000, "N_EU0")
            self.N_AS0 = Parameter(510, 100, 20000, "N_AS0")
            self.r_EU = Parameter(0.004, 0.0, 0.05, "r_EU")
            self.r_AS = Parameter(0.0055, 0.0, 0.05, "r_AS")
            self.T_AF = Parameter(8800, 8000, 15000, "T_AF")
            self.T_B = Parameter(5600, 2000, 8000, "T_B")
            self.T_EU_AS = Parameter(848, 100, 2000, "T_EU_AS")
            self.m_AF_B = Parameter(25e-5, 0.0, 0.01, "m_AF_B")
            self.m_AF_EU = Parameter(3e-5, 0.0,  0.01, "m_AF_EU")
            self.m_AF_AS = Parameter(1.9e-5, 0.0, 0.01, "m_AF_AS")
            self.m_EU_AS = Parameter(9.6e-5, 0.0, 0.01, "m_EU_AS")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

            '''
            # additional params for OOA (dadi) TODO where did these come from?
            self.m_AF_B = Parameter(None, 0, 50, "m_AF_B")   # e-5
            self.m_AF_EU = Parameter(None, 0, 50, "m_AF_EU") # e-5
            self.m_AF_AS = Parameter(None, 0, 50, "m_AF_AS") # e-5
            self.m_EU_AS = Parameter(None, 0, 50, "m_EU_AS") # e-5
            '''

        # bndx (ABC-DLS BNDX priors)
        elif simulator == simulation.bndx:
            self.N_A = Parameter(None, 5000, 25000, "N_A")
            self.N_AF = Parameter(None, 10000, 150000, "N_AF")
            self.N_EU = Parameter(None, 10000, 150000, "N_EU")
            self.N_AS = Parameter(None, 10000, 150000, "N_AS")
            self.N_EU0 = Parameter(None, 500, 5000, "N_EU0")
            self.N_AS0 = Parameter(None, 500, 5000, "N_AS0")
            self.N_B = Parameter(None, 500, 5000, "N_B")
            self.N_BC = Parameter(None, 500, 40000, "N_BC")
            self.N_AF0 = Parameter(None, 500, 40000, "N_AF0")
            self.N_MX = Parameter(None, 500, 40000, "N_MX")
            self.N_B0 = Parameter(None, 500, 40000, "N_B0")

            # times in ky
            self.T_FM = Parameter(None, 2, 5, "T_FM")
            self.T_FS = Parameter(None, 0.1, 10, "T_FS")
            self.T_DM = Parameter(None, 10, 50, "T_DM")
            self.T_EU_AS = Parameter(None, 5, 80, "T_EU_AS") # OOA [15-80], BNDX [5-30]
            self.T_NM = Parameter(None, 5, 50, "T_NM")
            self.T_XM = Parameter(None, 5, 120, "T_XM")
            self.T_Mix = Parameter(None, 5, 50, "T_Mix")
            self.T_Sep = Parameter(None, 5, 50, "T_Sep")
            self.T_B = Parameter(None, 5, 320, "T_B") # OOA [5-320], BNDX [5, 270]
            self.T_AF = Parameter(None, 5, 700, "T_AF")
            self.T_N_D = Parameter(None, 330, 450, "T_N_D")
            self.T_H_A = Parameter(None, 120, 250, "T_H_A")
            self.T_H_X = Parameter(None, 450, 700, "T_H_X")

            # percentages
            self.Mix = Parameter(None, 5, 95, "Mix")
            self.NMix = Parameter(None, 1, 3, "NMix")
            self.DMix = Parameter(None, 0, 2, "DMix")
            self.XMix = Parameter(None, 0, 10, "XMix")
            self.FMix = Parameter(None, 0, 10, "FMix")

            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # baboon
        elif simulator == simulation.baboon:
            # baboon params (in years, not ky)
            self.Tb_A = Parameter(1547656, 1100000, 3000000, "Tb_A")
            #self.Tb_B = Parameter(450, 100, 1000, "Tb_B")
            self.Tb_C = Parameter(88611, 10000, 100000, "Tb_C")
            self.Tb_D = Parameter(246445, 10000, 800000, "Tb_D")
            #self.Tb_E = Parameter(327631, 100000, 800000, "Tb_E")
            self.Tb_F = Parameter(152344, 10000, 800000, "Tb_F")
            self.Tb_G = Parameter(124713, 10000, 200000, "Tb_G")
            #self.Tb_H = Parameter(150, 100, 1000, "Tb_H") # no T_I
            self.Tb_J = Parameter(300000, 100000, 800000, "Tb_J")
            #self.Tb_K = Parameter(50, 100, 1000, "Tb_K")

            self.reco = Parameter(1e-8, 1e-9, 1e-7, "reco") # TODO made this up
            self.mut = Parameter(0.9e-8, 1e-9, 1e-7, "mut") # from paper

        else:
            sys.exit(str(simulator) + " not supported")

    def update(self, names, values):
        """Based on generator proposal, update desired param values"""
        assert len(names) == len(values)

        for j in range(len(names)):
            param = names[j]

            # credit: Alex Pan (https://github.com/apanana/pg-gan)
            attr = getattr(self, param)
            if attr is None:
                sys.exit(param + " is not a recognized parameter.")
            else:
                attr.value = values[j]

    def event_times_bndx(self):
        """
        Based on the time intervals, compute event times from the current
        parameter values
        """
        E_DM = round(self.T_DM.value)
        E_EU_AS = round(self.T_DM.value + self.T_EU_AS.value)
        E_NM = round(E_EU_AS + self.T_NM.value)
        E_XM = round(E_EU_AS + self.T_XM.value)
        E_Mix = round(E_EU_AS + self.T_Mix.value)
        E_Sep = round(E_Mix + self.T_Sep.value)
        E_B = round(max(E_NM, E_XM, E_Sep) + self.T_B.value)
        E_AF = round(E_B + self.T_AF.value)
        E_N_D = round(self.T_N_D.value)
        E_H_A = round(self.T_N_D.value + self.T_H_A.value)
        E_H_X = round(self.T_H_X.value)

        # 11 events total
        print("{:>8} Denisovan admixture into ASN".format(E_DM))             # 1
        print("{:>8} EUR/ASN split".format(E_EU_AS))                         # 2
        print("{:>8} Neanderthal admixture into OOA".format(E_NM))           # 3
        print("{:>8} African archaic admixture into AFR".format(E_XM))       # 4
        print("{:>8} Back to Africa admixture".format(E_Mix))                # 5
        print("{:>8} Separation of Back to Africa pop and OOA".format(E_Sep))# 6
        print("{:>8} Separation of African and OOA".format(E_B))             # 7
        print("{:>8} Decrease in African pop size".format(E_AF))             # 8
        print("{:>8} Neanderthal/Denisovan split".format(E_N_D))             # 9
        print("{:>8} Archaic/Human split".format(E_H_A))                     #10
        print("{:>8} African archaic/Human split".format(E_H_X))             #11

    def event_times_baboon(self):
        # these events correpond to the figure from the paper
        E_G = round(self.Tb_G.value)
        E_J = round(self.Tb_J.value)
        E_D = round(E_G + self.Tb_D.value)
        E_C = round(E_D + self.Tb_C.value)
        E_F = round(E_J + self.Tb_F.value)
        E_A = round(E_F + self.Tb_A.value)

        print("{:>8} G".format(E_G))
        print("{:>8} J".format(E_J))
        print("{:>8} D".format(E_D))
        print("{:>8} C".format(E_C))
        print("{:>8} F".format(E_F))
        print("{:>8} A".format(E_A))
