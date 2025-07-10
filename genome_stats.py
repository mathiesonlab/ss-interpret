"""
This version computes summary statistics for regions of real data along the
genome.
Authors: Rebecca Riley, Sara Mathieson
Date: 12/14/22
"""

# python imports
import numpy as np
import sys

# our imports
from pg_gan import global_vars
from pg_gan import real_data_random
from pg_gan import ss_helpers
import ss_extra

# globals
#NUM_REGIONS = 100000
NUM_SNPS = global_vars.NUM_SNPS

def get_iterator(input_file, bed_file):
    iterator = real_data_random.RealDataRandomIterator(input_file, 
        global_vars.DEFAULT_SEED, bed_file)
    return iterator

################################################################################
# SUMMARY STATISTICS
################################################################################

def stats_along_genome(iterator, output_file):

    # options for summary stats (neg1 should be True for discriminator)
    neg1 = False
    region_len = False
    prev_chrom = None

    # setup output array
    all_stats = []

    # go through entire genome
    final_end = iterator.num_snps-NUM_SNPS
    num_total = 0
    #final_end = NUM_REGIONS*NUM_SNPS # fewer for testing
    for start_idx in range(0, final_end, NUM_SNPS):
        curr_chrom = iterator.chrom_all[start_idx]
        if curr_chrom != prev_chrom:
            print("chrom", curr_chrom, "idx", start_idx)
            prev_chrom = curr_chrom

        # get the region of real data
        region = iterator.real_region(neg1, region_len, start_idx=start_idx)

        # compute summary stats
        if region is not None:
            corrected = np.zeros((1, iterator.num_samples, NUM_SNPS, 2))
            corrected[0] = region

            # TODO double check "max_dist" in ss_helpers
            stats = ss_helpers.stats_all(corrected)
            stats_extra = ss_extra.compute_extra_stats(region)
            all_stats.append(stats_extra) # TODO check dims

        num_total += 1

    print("num good regions", len(all_stats), "/", num_total) #NUM_REGIONS)
    np.save(output_file, np.array(all_stats))

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    h5_filename = sys.argv[1]
    bed_filename = sys.argv[2]
    output_file = sys.argv[3]

    # get iterator which will return real genomic regions
    iterator = get_iterator(h5_filename, bed_filename)

    # statistics for all regions
    stats_along_genome(iterator, output_file)
