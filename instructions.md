## how to make this work

# 1. Preprocessing
Run  preprocessing.py and preprocessing_spatial_data.ipynb

If you really want run review_sentiment_multilingual - will probably take very long depending on machine - took me 3 hours. 
Results of this skript can be found in data (listing_sentiment_multilingual.csv)

combined_data.csv ist our central file for training and testing

# 2. try the models
 - castboost.ipynb - with hyperparamter analysis
 - castboost_correction.ipynb - we tried to fit the high prices better, as they are estimeated lower than actaul 
 - Randomforest.ipynb
 - model_xgboost.ipynb
 - knn_spatial.ipynb - only trained with the spatial data
