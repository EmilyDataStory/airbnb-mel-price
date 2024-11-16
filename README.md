# Predicting Rental Prices 
- A data science project for building predictive models for Airbnb prices in Melbourne
Here’s a directory for your blog with links to the sections:

---

## Directory

- [Predicting Rental Prices](#predicting-rental-prices)  
- [Project Overview](#project-overview)  
- [Business Context](#business-context)  
- [Dataset](#dataset)  
  - [Categories of Data](#categories-of-data)  
  - [Data Types](#data-types)  
  - [Additional Data - Neighbourhood Boundaries](#additional-data---neighbourhood-boundaries)  
- [Data Preparation and Analysis](#data-preparation-and-analysis)  
  - [Data Cleaning](#data-cleaning)  
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Feature Engineering](#feature-engineering)  
- [Feature Selection](#feature-selection)  
- [Model Training and Evaluation](#model-training-and-evaluation)  
  - [Results](#results)  
  - [Top Features](#top-features)  
  - [Actionable Insights for Stakeholders](#actionable-insights-for-stakeholders)  
- [Improvement / Future Development](#improvement--future-development)  
- [Conclusion](#conclusion)  
- [Repository Contents](#repository-contents)  
- [About the Author](#about-the-author)

---

You can add this directory to the top of your blog to help readers navigate easily!

## Project Overview
This project builds a predictive model to estimate Airbnb prices in Melbourne, Australia, using data from September 2024. By analysing property features, location, host details, and listing information, we identify key features associated with price variation. 

This project is valuable not only for Airbnb hosts and real estate investors looking to optimise rental pricing but also for other domains like tourism and property management, where understanding pricing trends and influencing factors can support data-driven decision-making and strategic planning.

## Business Context
The short-term rental market, dominated by platforms like [Airbnb](https://airbnb.com), has grown significantly over the past decade. This growth has brought new challenges for hosts and property investors to set competitive prices that attract guests while maximising revenue. 

By leveraging data analysis and predictive models, hosts and investors can gain insights into the factors that correlate with price variation, helping them make more informed pricing decisions. For Airbnb, providing accurate pricing guidance can enhance user satisfaction and maintain its competitive edge in the evolving rental market.

## Dataset

This project uses Airbnb data sourced from [Inside Airbnb](https://insideairbnb.com/), which provides publicly available data on Airbnb listings. The dataset was scraped on **6th September 2024** and includes detailed information about Airbnb properties in Melbourne.
![image](https://github.com/user-attachments/assets/d4fc764a-4d42-44ee-943f-d155d75572e9)

- **Total Size:** 25,089 rows x 75 columns
- **Target Variable:** price (AUD per night)

### Categories of Data:
- **Property Information:** Includes details like property type, room type, amenities, and number of bedrooms
- **Location Information:** Latitude and longitude coordinates
- **Host Information:** Information about the host such as Superhost status, total listings, and response rate.
- **Listing Information:** Includes price, availability, number of reviews, and other listing details.

### Data Types:
- **Numeric:** Examples include accommodates, number of reviews, and latitude/longitude.
- **Categorical:** Examples include room type, neighbourhood, and host status.
- **Text:** Descriptive fields like neighbourhood overview and amenities.

### Additional Data - Neighbourhood Boundaries: 
- A geojson file was used to provide geographic boundaries of neighbourhoods within Melbourne, allowing for detailed spatial analysis and visualisation on maps.

This comprehensive dataset provides a foundation for analysing various factors associated with Airbnb pricing, allowing for in-depth insights into trends and key influences within Melbourne’s short-term rental market.

## Data Preparation and Analysis

In this section, we perform initial data exploration, handle missing values, clean and preprocess the data, and visualise key features to understand patterns and relationships with the target variable (price). This process sets the foundation for building an effective predictive model.

### Data Cleaning

1. **Missing Values**:
   - For columns with over 30% missing values (e.g., `neighbourhood_group_cleansed` and `calendar_updated`), these were dropped as they were deemed irrelevant or not usable for our analysis.
   - Binary columns (e.g., `host_is_superhost`, `host_identity_verified`) were cleaned by filling nulls with 0.
   - For important features such as `bedrooms`, `bathrooms`, and `price`, where values were missing, we used specific methods:
     - **Bedrooms**: Imputed based on the maximum guests (`accommodates`).
     - **Bathrooms**: Extracted from the `bathrooms_text` column where possible.
     - **Price**: Rows with missing prices were dropped, as this was our target variable.
  
2. **Outliers**:
   - Extreme outliers in the `price` column were identified and capped at the 99th percentile to avoid skewing the model.
   - For visualisation purposes, log transformation was applied to `price` to achieve a more normalised distribution.

### Exploratory Data Analysis (EDA)

1. **Target Variable - Price**:
   - Price distribution was initially highly skewed. After capping and log transformation, the distribution became more normal, making it more suitable for modelling.
![image](https://github.com/user-attachments/assets/5b300cb0-55c8-4be9-a7ad-34873166a33c)

2. **Visualising Key Features**:
   - **Number of Bedrooms**: A violin plot of `price` by `bedrooms` showed a clear trend where properties with more bedrooms generally had higher prices.
   ![image](https://github.com/user-attachments/assets/f91952d4-41c7-40da-9c9e-29cf1751540e)

   - **Room Type**: A similar analysis showed that entire homes/apartments commanded higher prices compared to private or shared rooms.
   ![image](https://github.com/user-attachments/assets/1301bab8-9cc9-4b1c-a8e4-6ca63a6901e7)


   - **Location (Latitude and Longitude)**: A scatter plot of location by price demonstrated how certain areas had higher concentrations of high-priced listings, allowing us to see spatial trends.
![image](https://github.com/user-attachments/assets/e26b3289-f9b5-4b8e-bbc4-0cea49692a48)

  - (Median Prices by Neighbourhood):
    - Colour codes: red for the highest median prices (above $250), yellow for mid-high prices (between $150 and $250), green for mid-range prices (between $100 and $150), and blue for the lowest median prices (below $100)
  ![image](https://github.com/user-attachments/assets/f3f71c51-e45f-4388-8649-a2b1e8966731)

3. **Feature Correlations**:
   - We examined correlations between numeric variables and price, particularly focusing on features like `accommodates`, `bedrooms`, and `distance_to`, which showed moderate to strong associations with price.
   - Multicollinearity was assessed to avoid redundancy in the model, particularly for features like `availability_30` and `availability_60`. When highly correlated, only the most relevant feature was retained, unless they are relevant to different aspects of pricing (e.g. `beds_num_cap` and `accommodates`).
![image](https://github.com/user-attachments/assets/094cc42a-7254-4f8e-ad00-e926c759e9d9)
     
### Feature Engineering
1. **Creating New Features**:
   - **Amenities Count**: Converted the amenities list to a count of amenities, giving a numeric feature for each listing.
   - **Distance to Key Locations**: Calculated the distance from each listing to the Melbourne city centre and major landmarks (like the airport).
   - **Lat-Long Interaction**: Created an interaction term for latitude and longitude to capture spatial effects in pricing.
   - **Bathroom Privacy**: Added indicators for shared and private bathrooms based on keywords in the bathrooms_text column (is_shared_bath, is_private_bath).

2. **Handling Text and Categorical Data**:
   - Categorical features (e.g., `room_type`, `property_type`) were converted to dummy variables for model training.
   - For features like `amenities`, additional columns were created for specific amenities (e.g., `has_free_parking`, `has_self_checkin`) based on keyword search.
![image](https://github.com/user-attachments/assets/16b98b41-0fac-4c70-9060-d1bff0f26ef0)
![image](https://github.com/user-attachments/assets/4e0bae78-2fac-43f5-87ea-069b583fd7b3)

---

This EDA and data cleaning section not only prepares the data for modelling but also provides a comprehensive overview of the dataset, allowing us to understand key pricing patterns and trends within Melbourne’s Airbnb market.

## Feature Selection
Feature selection in this project was guided by several factors:

- **Business Understanding**: Based on literature and personal experience in property investment.
- **Correlation with Price**: While considered, it wasn’t a strict criterion due to potential non-linear relationships.
- **Feasibility for New Hosts**: Chose features that are practical and actionable for new hosts and investors.
- **Model-based Refinement**: After baseline model training, the final selection focused on the top 25 features identified through feature importance.

*Note*: Review scores were excluded as predictors. Instead, the dataset was filtered to include only properties with ratings above 3.9. This ensures the model reflects listings with actual guest experiences, making the predictions more relevant for optimising long-term pricing strategies.

## Model Training and Evaluation
1. Models Used:

- **XGBoost** and **LightGBM** were chosen as primary models for their effectiveness in handling structured data and capturing non-linear relationships.
- **K-Nearest Neighbours (KNN)** and **Linear Regression** were also included for comparison, providing insight into the performance of simpler models.

2. Data Splitting and Cross-Validation:

- The dataset was split into 80:20 training and test sets to evaluate model performance on unseen data.
- Additionally, 5-fold cross-validation (CV) was applied (on training set only) during hyperparameter tuning to ensure model robustness and reduce overfitting.

3. Hyperparameter Tuning:

- A randomised search with cross-validation was conducted for each model to optimise key parameters like learning rate, tree depth, and regularisation terms. 

4. Evaluation Metrics:

- **Adjusted R²** was calculated for both the log-transformed prices (used during training) and the original price scale (for real-world interpretability). It was selected as the primary metric over R² to account for the number of features in the model, providing a fairer comparison as complexity increased.
- Additional metrics included Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), each offering different perspectives on model accuracy and error distribution.

```python

## Hyperparameter Tuning for top models (based on Adjusted R² results and training time)

# Define parameter spaces for each model
lightgbm_params = {
    'num_leaves': [31, 64, 128, 256, 512],
    'max_depth': [-1, 10, 20, 30, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 700, 1000], 
    'min_data_in_leaf': [5, 10, 20],
    'min_child_weight': [1e-3, 0.01, 0.1, 1],
    'max_bin': [255, 300, 500, 1000],
    'lambda_l1': [0, 0.1, 1, 5],
    'lambda_l2': [0, 0.1, 1, 5],
    'min_split_gain': [0, 0.1, 0.5, 1],
    'boosting_type': ['gbdt', 'dart', 'goss'],
}

xgboost_params = {
    'max_depth': [3, 5, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 700, 1000],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

knn_params = {
    'n_neighbors': [3, 5, 15, 25],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


# Define the models and parameter grids
models_info = [
    # Models that do not require scaling (tree-based models)
    ('LightGBM', LGBMRegressor(random_state=seed, verbose=0), lightgbm_params, False),
    ('XGBoost', XGBRegressor(random_state=seed), xgboost_params, False),

    # Models that require scaling (linear models)
    ('Linear Regression', LinearRegression(), {}, True),
    ('KNN', KNeighborsRegressor(), knn_params, True)
]

# Define the cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# Initialize lists to store results
results_list = []
best_estimators = {}
best_params = {}
all_predictions = {}

# Use the selected features
X_train = X_train[selected_features]
X_test = X_test[selected_features]

for model_name, model, params, needs_scaling in models_info:
    # Create a pipeline with scaling if necessary
    if needs_scaling:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        # Adjust parameter grid to match the pipeline structure
        pipeline_params = {
            f'model__{key}': value for key, value in params.items()
        }
    else:
        pipeline = Pipeline([
            ('model', model)
        ])
        pipeline_params = {
            f'model__{key}': value for key, value in params.items()
        }

    # Perform RandomizedSearchCV with the pipeline
    random_search = RandomizedSearchCV(
        pipeline, pipeline_params, n_iter=50, cv=kf,
        scoring='neg_mean_absolute_error', n_jobs=-1, random_state=seed
    )

    # Fit the model and record training time
    start_time = time.time()
    random_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Get the best estimator and parameters
    best_estimator = random_search.best_estimator_
    best_param = random_search.best_params_

    # Store best estimator and parameters
    best_estimators[model_name] = best_estimator
    best_params[model_name] = best_param

    # Record prediction time
    start_time = time.time()
    y_pred_log = best_estimator.predict(X_test)
    prediction_time = time.time() - start_time

    # Exponentiate predictions and actuals to get original scale
    y_test_exp = np.exp(y_test)
    # Bias correction when exponentiating predictions
    sigma_squared = np.var(y_test - y_pred_log)
    y_pred_exp = np.exp(y_pred_log + sigma_squared / 2)

    n = len(y_test)  # number of samples
    p = X_test.shape[1]  # number of features

    # Compute evaluation metrics on log scale
    r2_log = r2_score(y_test, y_pred_log)
    adj_r2_log = adjusted_r2(r2_log, n, p)  # 1 - (1 - r2_log) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    # Compute evaluation metrics on original scale
    r2 = r2_score(y_test_exp, y_pred_exp)
    adj_r2 = adjusted_r2(r2, n, p)  # 1 - (1 - r2) * (len(y_test_exp) - 1) / (len(y_test_exp) - X_test.shape[1] - 1)
    mse = mean_squared_error(y_test_exp, y_pred_exp)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    mape = mean_absolute_percentage_error(y_test_exp, y_pred_exp) * 100  # MAPE in percentage

    # Append results with specified precision
    results_list.append({
        'Model': model_name,
        'Adjusted R²_Log': round(adj_r2_log, 3),
        'Adjusted R²': round(adj_r2, 3),
        'MAPE (%)': round(mape, 2),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'Training Time': round(training_time, 2),
        'Prediction Time': round(prediction_time, 3)
    })

    # Store predictions for further analysis if needed
    all_predictions[model_name] = {
        'y_pred_log': y_pred_log,
        'y_pred_exp': y_pred_exp,
        'y_test_exp': y_test_exp
    }

# Create a DataFrame from the results
final_results_df = pd.DataFrame(results_list)

# Reorder columns
final_results_df = final_results_df[[
    'Model', 'Adjusted R²_Log', 'Adjusted R²', 'MAPE (%)', 'MAE', 'RMSE',
    'Training Time', 'Prediction Time'
]]

```

## Results:
![image](https://github.com/user-attachments/assets/e01ee80b-dbef-4708-b658-cc70208fe1ca)
- Top Model: **XGBoost** achieved the highest performance with an Adjusted R² of 0.674 on the original price scale and 0.777 on the log-transformed scale. Its MAPE of 21.87% indicates that predictions are, on average, within 22% of the actual prices.
- Other Models: LightGBM followed closely with similar performance, while KNN and Linear Regression lagged behind, with lower adjusted R² and higher errors.
- Training and Prediction Times: XGBoost demonstrated a good balance of performance and efficiency

### Plot that compares the predicted prices from the XGBoost model to actual prices
![image](https://github.com/user-attachments/assets/08af155e-0178-487b-8ef0-a50a851186ab)
- Accurate for Mid-Range Prices: The model predicts prices well for listings between 100–500 AUD, aligning closely with actual values.
- Challenges with High-End Listings: Underestimation occurs for properties priced above 600 AUD, indicating unique factors not captured by the model.
- Useful for Hosts: The model provides actionable insights for most listings, helping optimise pricing for the majority market segment.

### Top Features:
- The top 10 features contributing to XGBoost’s prediction were:
![image](https://github.com/user-attachments/assets/fedd7544-d044-47ca-91e9-0bfb7e65f90a)

- The top 10 features contributing to LightGBM’s prediction were:
![image](https://github.com/user-attachments/assets/217b8748-5604-4a0d-ad36-a98bc83462b4)

### Actionable Insights for Stakeholders:
- New Hosts/Investors: Prioritise private amenities and choose room/property types that are in demand (XGBoost insights).
- Experienced Hosts: Enhance amenities and availability while leveraging reviews and operational stability to maintain competitive pricing (LightGBM insights).

## Improvement / Future Development

1. **Incorporate Seasonality**: Include seasonal and booking time trends to better capture price fluctuations throughout the year.  
2. **Add More Location Context**: Leverage neighbourhood-level data or proximity to specific attractions to improve spatial understanding.  
3. **Refine High-End Predictions**: Address underperformance for high-priced listings by exploring luxury-specific features or additional data sources.  

## Conclusion

This project successfully developed a predictive model for Airbnb prices in Melbourne, achieving moderate to strong performance metrics. The findings provide valuable insights for hosts and investors to optimise pricing strategies. While there’s room for improvement, the model offers a solid foundation for data-driven decision-making in the short-term rental market.

---
## Repository Contents

- **Jupyter Notebooks**:  
  - `airbnb_mel_01_preprocess.ipynb`  
  - `airbnb_mel_02_model.ipynb`  

- **Datasets**:  
  - **Original Files**:  
    - `ab_mel_listings_2024.csv`  
    - `ab_mel_neighbourhoods.geojson`  
  - **Processed File**:  
    - `airbnb_mel_log_prep_02.csv`  

---
## About the Author
Emily Huang - Data Science enthusiast with a passion for turning data into actionable insights.

Feel free to connect with me via my [LinkedIn Profile](https://www.linkedin.com/in/emily-huang-3021212a)
  
