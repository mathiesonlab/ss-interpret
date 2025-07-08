"""
Description: read combined hgdp+1kgp data, split into populations, infer the
    ARG, and calculate eGRM on sliding windows (window size set to 50kb, step
    size to 10kb)
Usage: python3 gnomad_relate.py IN_FOLDER OUT_FOLDER
Author: Jordan Cahoon, Sara Mathieson
Date: 6/12/24

TODO:
- adjust relate simulation
- create tarball for input into model
"""

# python imports
import numpy as np
from subprocess import Popen, PIPE
import sys

# our imports
from pg_gan import real_data_random

################################################################################
# GLOBALS
################################################################################

# TODO expand to all pops
#CHR = "21"
POP = "ESN"

IN_FOLDER = sys.argv[1]
BED_FILE = sys.argv[2]
OUT_FOLDER = sys.argv[3] + "/" + POP
#mkdir = Popen("mkdir " + OUT_FOLDER, shell=True, stdout=PIPE)
#mkdir.communicate()

WINDOW = 50000 # 50kb # TODO windows should be by SNPs!
STEP   = 50000 # non-overlapping
#MIN_SNPS = 50 # min SNPs per 50kb region

################################################################################
# HELPERS
################################################################################

def read_chrom_lengths():
    # TODO should be hg37 but probably close enough
    arr = np.loadtxt("hg38_chrom_lengths.tsv", dtype='int', delimiter="\t", skiprows=1)
    chrom_dict = {}
    for chr in range(1,23):
        assert arr[chr-1][0] == chr
        chrom_dict[str(chr)] = arr[chr-1][1]
    return chrom_dict

################################################################################
# MAIN
################################################################################

def one_chrom(chrom, chrom_dict, mask_dict):

    # length of chrom
    chrom_length = chrom_dict[chrom]

    # index
    vcf_filename = IN_FOLDER + "/" + POP + ".chr" + chrom + ".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
    index = "bcftools index " + vcf_filename
    process = Popen(index, shell=True, stdout=PIPE)
    process.communicate()

    # go through each region
    start = 0
    end = WINDOW
    kept = 0
    total = 0
    while end <= chrom_length:
        total += 1

        cmd = "bcftools view --no-header -r chr" + chrom + ":" + str(start) + "-" + str(end) + " " + vcf_filename + " | wc -l"
        #print(cmd)
        process = Popen(cmd, shell=True, stdout=PIPE)
        output, err = process.communicate()
        num_snps = int(output.decode("utf-8"))
        #input('enter')

        # create region to determine accessibility
        region = real_data_random.Region(chrom, start, end)
        
        # if we have enough SNPs and inside accessibility mask
        if region.inside_mask(mask_dict):
            kept += 1

            #print("num SNPs", num_snps)
            prefix = POP + "_chr" + chrom + "_" + str(start) + "_" + str(end)

            # extract SNPs
            bcftools_cmd = "bcftools view -r chr" + chrom + ":" + str(start) + "-" + str(end) + " -Oz -o " + prefix + ".vcf.gz " + IN_FOLDER + "/" + POP + "/" + POP + "_chr" + chrom + ".vcf.gz"
            #process = Popen(bcftools_cmd, shell=True, stdout=PIPE)
            #process.communicate() # wait to finish

            # convert from vcf.gz to haps and sample
            relate = "RelateFileFormats --mode ConvertFromVcf --haps " + prefix + ".haps --sample " + prefix + ".sample -i " + prefix
            #process = Popen(relate, shell=True, stdout=PIPE)
            #process.communicate() # wait to finish

            # infer tree (biallelic snps retained with bcftools))
            relate = "Relate --mode All -m 1.25e-8 -N 20000 --haps " + prefix + ".haps --sample " + prefix + ".sample --map ../simulation/genetic_map.txt --seed 1 -o " + prefix
            #process = Popen(relate, shell=True, stdout=PIPE)
            #process.communicate() # wait to finish
            
            # convert to tree sequence
            relate = "RelateFileFormats --mode ConvertToTreeSequence -i " + prefix + " -o " + prefix + ".infer"
            #process = Popen(relate, shell=True, stdout=PIPE)
            #process.communicate() # wait to finish

            # calculate GRM on entire region
            '''egrm = "trees2egrm --output-format numpy " + prefix + ".infer.trees --c --haploid --output " + OUT_FOLDER + "/" + prefix #" --left ${win_start} --right ${win_end}
            process = Popen(egrm, shell=True, stdout=PIPE)
            process.communicate() # wait to finish'''

            # clean 
            clean = "rm -rf " + prefix + "*"
            process = Popen(clean, shell=True, stdout=PIPE)
            process.communicate() # wait to finish

        start += STEP
        end += STEP

    print("frac kept", kept/total, "num kept", kept)
    return kept

# TODO: tarball of PATH_TO_OUTPUT/*.npy files

if __name__ == "__main__":
    chrom_dict = read_chrom_lengths()
    # accessibility mask
    mask_dict = real_data_random.read_mask(BED_FILE)

    total_regions = 0
    for chrom in range(1,23):
        num_regions = one_chrom(str(chrom), chrom_dict, mask_dict)
        print("chrom", chrom, "num regions", num_regions)
        total_regions += num_regions
    
    print("total regions", total_regions)

