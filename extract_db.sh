#!/bin/bash

REPO_DIR=/home/brewst27/projects/def-moosavi5/brewst27/CGCNN_MOF

cd $REPO_DIR


# Create the target directory if it doesn't exist
mkdir -p test_datasets/CoRE2019

echo "Extracting CoRE2019 dataset to test_datasets/CoRE2019"
# Extract the tar file to the target directory
tar --strip-components=1 -xf test_datasets/CoRE2019.tar -C test_datasets/CoRE2019

echo "CoRE2019 dataset extracted successfully to test_datasets/CoRE2019"