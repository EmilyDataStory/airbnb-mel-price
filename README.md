# Predicting Rental Prices 
- A data science project for building predictive models for Airbnb prices in Melbourne

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
