# Wine Quality Prediction

## Table of Contents  
- [Description](#description)  
- [Installation](#installation)  
- [Usage](#usage) 
- [Repo files](#files) 

---

## Description  
This repository contains a workflow for predicting the quality of a wine based on its chemical properties. </br>

It includes code for data preprocessing, feature engineering, and developing a **Neural network** model for wine quality prediction.  

This project explores the use of neural networks for regression analysis. While it can also be approached as a classification task, the class distribution is highly imbalanced.

---

## Installation  
To run this project locally, follow these steps:  
```bash
git clone https://github.com/dha-code/Wine_quality_prediction.git
cd Wine_quality_prediction  
```
---

## Usage

The code uses the [Wine dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data) downloaded from Kaggle (also present within the ./data folder). The ./figures folder has the graphs generated by the code

   ```python
   python main.py
   ```
  
---

## Timeline

This project was developed in one week from 28/01/2025 - 03/02/2025

---

## Files 
      
* [data/](./Wine_quality_prediction/data)
  * [TVsynthetic_wine_data.csv](./Wine_quality_prediction/data/TVsynthetic_wine_data.csv)
  * [WineQT.csv](./Wine_quality_prediction/data/WineQT.csv)
  * [WinesCleaned.csv](./Wine_quality_prediction/data/WinesCleaned.csv)
* [figures/](./Wine_quality_prediction/figures)
  * [Cleandata_Boxplots.png](./Wine_quality_prediction/figures/Cleandata_Boxplots.png)
  * [Cleandata_Clustermap.png](./Wine_quality_prediction/figures/Cleandata_Clustermap.png)
  * [Rawdata_Boxplots.png](./Wine_quality_prediction/figures/Rawdata_Boxplots.png)
  * [Rawdata_Clustermap.png](./Wine_quality_prediction/figures/Rawdata_Clustermap.png)
  * [Train_vs_test.png](./Wine_quality_prediction/figures/Train_vs_test.png)
* [utils/](./Wine_quality_prediction/utils)
  * [eda.py](./Wine_quality_prediction/utils/eda.py)
  * [model.py](./Wine_quality_prediction/utils/model.py)
  * [model_metrics.py](./Wine_quality_prediction/utils/model_metrics.py)
  * [__init__.py](./Wine_quality_prediction/utils/__init__.py)
* [main.py](./Wine_quality_prediction/main.py)
