# Airbnb Price Prediction Project
### Tutorial 1, Group 7: Minh Tran, Philip Dörnte, Clara Bläser
### Framework
1. **City**: Sevilla, Andalucía, Spain
2. **Downloaded Data**: listing, calendar, reviews
3. **Modalities**:
   - **Tabular Data**: listing and calendar
   - **Text Data**: reviews
4. **modeling approaches**: (Compare and combine different modeling approaches)
   - Random forest


   - Linear or polynomial regression
   - Random forests or gradient-boosted trees
   - Neural networks (e.g., multi-layer perceptron, convolutional networks, transformers)
   - Hybrid models (e.g., text embeddings from a language model combined with a regression or tree-based model)
5. **Evaluation**: Split your dataset into training and testing subsets. Evaluate models using appropriate metrics and compare their performance.


## 1 Data Cleaning & Preparation
### 1.1 listing
1. Cleaned price by removing currency symbols, converting it to numeric, removing price outliers (top and bottom 1%) and droping rows with missing prices
2. Select a subset of columns for modeling: "price",
    "room_type", "property_type", "accommodates", "bathrooms", "bedrooms", "beds", "neighbourhood_cleansed", "latitude", "longitude", "minimum_nights", "maximum_nights", "instant_bookable", "availability_365", "number_of_reviews", "review_scores_rating", "review_scores_cleanliness", "review_scores_location", "review_scores_value", "reviews_per_month", "host_is_superhost", "host_response_rate", "host_total_listings_count", "name", "description", "amenities"
3. Identify object‑type columns and count their unique values
4. cleaned the host_response_rate column by removing '%' and converting it to numeric
5. Imputeed missing host_response_rate values using the median.
6. Detect missing values in key categorical columns.
7. Imputeed missing host_is_superhost values using the most frequent category.
8. Droped rows with missing bathrooms, bedrooms, or host_total_listings_count
9. Filled remaining numeric review‑related NaNs using median imputation.
10. Replaced missing text fields (name, description, amenities) with empty strings.

normalize or scale numerical values??



## 2. **Feature Engineering**  
### 2.1 listing
1.  Converted binary categorical fields (instant_bookable, host_is_superhost) into 0/1 numeric values.
2. One‑Hot Encoding for room_type and merge resulting columns
3. Labeled Encoding for property_type and  neighbourhood_cleansed

### 2.2 calendar
1. calculated booked days
2. calculated booked week- and weekenddays
3. calculated booked days for each season

### 2.3 reviews
...

   - Encode categorical variables (e.g., one-hot encoding for neighborhood).  
   - Extract text features (e.g., TF-IDF, pretrained embeddings,...).  
   - Process images (e.g., resize, extract embeddings,...).  
## 3. **Modeling & Tuning**  
   - Model creation
   - Experiment with hyperparameter tuning.  
   - Evaluate models on a held-out dataset to assess generalization.
## **Analysis & Reporting**  
   - Document your findings with clear visualizations (plots, tables).  
   - Discuss trade-offs between models, modalities, and feature choices.  
   - Suggest possible improvements.

