# KnowMDD
This is the Pytorch implementation of KnowMDD in the paper: [KnowMDD: Knowledge-guided Cross Contrastive Learning for MDD Diagnosis].

## Requirements
- pytorch
- pytorch-geometric
- pandas
- nilearn


## Usage
To run the code
1. download the datasets REST-meta-MDD Project from DIRECT Consortium, which you can download from: https://www.scidb.cn/en/detail?dataSetId=cbeb3c7124bf47a6af7b3236a3aaf3a8.
2. place the downloaded data into a new 'data' folder
1. run the Make_FC_data.py
2. run the main.py

## To handle the raw datasets
1. download the datasets REST-meta-MDD
2. find the `ROISignals_FunImgARCWF` folder and the 'REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx' and place them in the 'data' folder
