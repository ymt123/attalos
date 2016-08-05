# YFCC Data Processing

The YFCC 100million dataset's size requires some special processing. The starting assumption is that you've downloaded the metadata files describing the colleciton located at https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67
You will need two sets of files:
*  yfcc100m_dataset-X.bz2: Files containing the metadata for the yfcc collection (X is a number from 0 through 9)
*  yfcc100m_hash.bz2: Maps the hash values used in the files below to the Ids in the metadata

This also assumes that you've downloaded the VGG features from (http://www.multimediacommons.org/getting-started) located at: http://multimedia-commons.s3-website-us-west-2.amazonaws.com/?prefix=features/image/vgg-vlad-yfcc/vgg/
The files are named "yfcc100m_dataset-X.tar.gz" where X is a number from 0 through 9


Preparing the curated data is a multistep processing:
1) Create the json file normally created by the attalos.preprocessing.text script
2) Create the hdf5 file normally created by the attalos.preprocessing.image scripts
3) Create the curated file using the create_curated.py file which takes a dataset iterator (requiring steps 1 and 2)