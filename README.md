# Kaggle_HomeDepot

110th Place Solution for [Home Depot Product Search Relevance][kaggle_home]

### Dependencies
- `scikit-learn`
- `pandas`
- `numpy`
- `pychant`
- `keras`
- `xgboostpan`

### FlowChart

### Configuration 
The configuration of the project can be changed from `configs.py`.

Note: Use `python3` as default interpreter.

### Regenerating Results
Before running any of the files copy the data to `../input/` folder and and clone this repository in scripts folder. So, the project structure should look like

```
Kaggle_HomeDepot
????input
    ?   train.csv
    ?   test.csv
    ?   sample_submission.csv
????scripts
    ?   README.md
    ?   <Files of this repository>
```

Now, To regenerate the results run these files mentioned below sequentially. 

#### Data Pre-Processing and Feature Extraction

- `feature_generater.py` - Clean Data and generates TF-IDF features
- `features_distance.py` - Generates distance and counting features
- `generate_dataset_svd50x3.py` - Complies all the individual feature and generates the processed dataset.

#### Machine Learning and Stacked Generalization

- `StackedGeneralization.py` - Train all the machine learning modules and stacks all the results to create submission.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [kaggle_home]: <https://www.kaggle.com/c/home-depot-product-search-relevance>
