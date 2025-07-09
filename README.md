# CGCNN MOF: CGCNN and Transformer Models for Property Prediction based on Structural Represenation

This is a fork of the implementation of ["MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction"](https://pubs.acs.org/doi/10.1021/jacs.2c11420). This work proposes a structure-agnostic deep learning method based on the Transformer model, named as <strong><em>MOFormer</em></strong>, for property predictions of MOFs. <strong><em>MOFormer</em></strong> takes a text string representation of MOF (MOFid) as input, thus circumventing the need of obtaining the 3D structure of a hypothetical MOF and accelerating the screening process.

This repository trains Crystal Graph Convolutional Networks (CGCNNs) and Transformer-based models on the CoRE2019 Metal-Organic Framework (MOF) dataset, using crystal structure files (CIFs) and MOFIDs (molecular identifiers), respectively. The models are tasked with predicting a range of geometry-dependent and chemistry-dependent material properties.

Model performance is evaluated using metrics such as Spearman Rank Correlation Coefficient (SRCC), MAE, and R², allowing a direct comparison of predictive accuracy across property types. This enables analysis of the disparity in model performance when predicting structural/geometric properties versus chemical adsorption and affinity properties, shedding light on how well each architecture captures different facets of material behaviour. 

Below is the citation for the original paper:

```
@article{doi:10.1021/jacs.2c11420,
    author = {Cao, Zhonglin and Magar, Rishikesh and Wang, Yuyang and Barati Farimani, Amir},
    title = {MOFormer: Self-Supervised Transformer Model for Metal–Organic Framework Property Prediction},
    journal = {Journal of the American Chemical Society},
    volume = {145},
    number = {5},
    pages = {2958-2967},
    year = {2023},
    doi = {10.1021/jacs.2c11420},
    URL = {https://doi.org/10.1021/jacs.2c11420}
}
```


## Getting Started

### Installation

#### SLURM / HPC Cluster Setup

```
git clone https://github.com/zakariyyabrewster/CGCNN_MOF
cd CGCNN_MOF

module load python/3.11.5

pip install virtualenv
virtualenv myenv
source myenv/bin/activate
(myenv) pip install -r requirements.txt
(myenv) deactivate
```

OR Set up conda environment and clone the github repo

```
# create a new environment
$ conda create -n myenv python=3.9
$ conda activate moformer
$ conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
$ conda install --channel conda-forge pymatgen
$ pip install transformers
$ conda install -c conda-forge tensorboard

# clone the source code of MOFormer
$ git clone https://github.com/zakariyyabrewster/CGCNN_MOF
$ cd CGCNN_MOF
```

### Dataset

All the data used in this work can be found in the `test_datasets` folder. If you use any data in this work, please cite the corresponding reference included in the Acknowledgement.

## Run the Model
In SLURM, start the training pipeline for CGCNN:
```
sbatch cgcnn_run.sh 
```
Train the Transformer with:
```
sbatch transformer_run.sh
```
See below for customization + other setup instructions.
### Creating Datasets / ID-Property Tables

For CGCNN training, if you are using the CoRE2019 DB, the structures can be found at `CoRE2019.tar`. The structures will be extracted into your repo using the `extract_db.sh` script within `cgcnn_run.sh`. If you are not using CoRE2019 and want to use another DB that stores .cif structure files in a similar format to CoRE2019, edit the `extract_db.sh` script to create your custom directories and extract your DB. Note that if you have cloned your repo and have already ran the `cgcnn_run.sh` script, the structures will stay extracted and you can delete the `bash extract_db.sh` line.   

For CGCNN training on a specific property tracked in your master feature labels dataset (for CoRE2019, data table
is `test_datasets/full_CoRE2019_alldata.csv`), edit the `label_dir` and `target_properties` dictionary in `gen_prop_id.py` to generate ID-property tables.

For Transformer training on a specific property tracked in your master feature labels dataset (for CoRE2019, data table
is `test_datasets/full_CoRE2019_alldata.csv`), given your MOFID DB is formatted like `test_datasets/core_mofid.smi`, edit the `smi_path`, `label_dir`, and `target_properties` dictionary in `mofid_coversion.py` to generate MOFID-property tables.


### Training / Fine-tuning

For CGCNN training, edit your model/optimizer/training pipeline configurations in `config_ft_cgcnn.yaml`. Before running `cgcnn_run.sh`, edit the script as such:
```
bash train_cgcnn.sh target_property
```
where `target_property` is the property name you initialized in your datasets. 

You may also pass a list of properties as such:
```
bash train_cgcnn.sh property1 property2 property3
```
To sequentially train a model on each property at once. Note that you may need to change the SBATCH params, such as --time and --mem depending on how many properties you decide to train in a single job.

For Trasnformer training, edit your model/optimizer/training pipeline configurations in `config_ft_transformer.yaml`. Before running `transformer_run.sh`, edit the script as such:
```
bash train_transformer.sh target_property
```
where `target_property` is the property anme you initialized in your datasets. 

You can pass a list of properties similarly to the CGCNN method above.

### Evaluation on Model Performance

For SRCC evaluation/ranking of model performance on your set of properties, use `prop_comp_cgcnn.py` and `prop_comp_trans.py` for CGCNN and Transfomer respectively. In the `load_test_results` function, edit the list of properties you would like to compare.

Visualizations / Graphs / Tables will be saved in `training_results/analysis` under filenames based on the YAML configuration files.

### K-fold Cross Validation

WIP

## Acknowledgement
- CGCNN: [Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) and [Code](https://github.com/txie-93/cgcnn)
- Barlow Twins: [Paper](https://arxiv.org/abs/2103.03230) and [Code](https://github.com/facebookresearch/barlowtwins)
- Crystal Twins: [Paper](https://www.nature.com/articles/s41524-022-00921-5) and [Code](https://github.com/RishikeshMagar/Crystal-Twins)
- MOFid: [Paper](https://pubs.acs.org/doi/full/10.1021/acs.cgd.9b01050) and [Code](https://github.com/snurr-group/mofid/tree/master)
