TRAIN_POP=${1}  # first command line arg (i.e. CEU)
TEST_POP=${2} # second command line arg (i.e. GBR)
SEL_TYPE=${3} # AI, Aug23, Over, Over2
DISC="discriminators_og/arch"
DATE="250331"

# predictions
echo "python3 genome_disc.py /bigdata/smathieson/1000g-share/HDF5/${TEST_POP}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 /bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed /homes/smathieson/Documents/pg_gan_interpret/${DISC}/${TRAIN_POP}/ /homes/smathieson/Documents/pg_gan_interpret/${DISC}/predictions/ ${DATE}"
#python3 genome_disc.py /bigdata/smathieson/1000g-share/HDF5/${TEST_POP}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 /bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed /homes/smathieson/Documents/pg_gan_interpret/${DISC}/${TRAIN_POP}/ /homes/smathieson/Documents/pg_gan_interpret/${DISC}/predictions/ ${DATE} ${SEL_TYPE}

# last hidden layer
#echo "python3 genome_disc.py /bigdata/smathieson/1000g-share/HDF5/${TEST_POP}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 /bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed /homes/smathieson/Documents/pg_gan_interpret/${DISC}/${TRAIN_POP}/ /homes/smathieson/Documents/pg_gan_interpret/${DISC}/hidden/ ${DATE} ${SEL_TYPE}"
#python3 genome_disc.py /bigdata/smathieson/1000g-share/HDF5/${TEST_POP}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 /bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed /homes/smathieson/Documents/pg_gan_interpret/${DISC}/${TRAIN_POP}/ /homes/smathieson/Documents/pg_gan_interpret/${DISC}/hidden/ ${DATE} ${SEL_TYPE}