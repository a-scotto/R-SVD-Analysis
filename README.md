# Reproductibility of numerical results related to randomized low-rank approximations.

## Goal

The following scripts allow you to reproduce the numerical results given in Section 5 of the following manuscript: 
"A general error analysis for randomized low-rank approximation methods" by Y. Diouane, S. Gürol, A. Scotto di Perrotolo and X. Vasseur [LINK TO BE GIVEN WHEN AVAILABLE]. 

## Generating the experimental data

The data can be simply (re)generated using python _data_generation.py_. The computed data are stored in two distinct JSON files, depending on which parameter varies (target rank or oversampling parameter). For convenience, these two files are given here (_Data_versus_rank.json_ and _Data_versus_samples.json_).

## Generating the figures

To generate all the figures of Section 5, simply use python _plots.py_. Please note that the script does not open the figures, it rather saves them as PDF files.
