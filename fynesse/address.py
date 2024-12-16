# This file contains code for suporting addressing questions in the data
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.linear_model import Ridge
from . import assess
from shapely.geometry import box, Point
import geopandas as gpd

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""
def train_and_test_ols(data, train_features, target_features, test_size=0.3, random_state=42):
    """
    Train and test a linear model using OLS with normalized features and proper feature names.

    Parameters:
        data (pd.DataFrame): DataFrame containing the features and target.
        train_features (list of str): List of feature columns used for training.
        target_features (list of str): Target column name (list with one element).
        test_size (float): Proportion of data to use as test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: OLS model, train R^2, test R^2, and test RMSE.
    """
    # Extract features and target
    X = data[train_features]
    y = data[target_features[0]]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize the features (standardization: mean=0, std=1)
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Add a constant for the intercept term and create DataFrames with proper column names
    train_feature_names = ["const"] + train_features
    X_train_const = pd.DataFrame(sm.add_constant(X_train_normalized, has_constant="add"),
                                 columns=train_feature_names,
                                 index=X_train.index)
    X_test_const = pd.DataFrame(sm.add_constant(X_test_normalized, has_constant="add"),
                                columns=train_feature_names,
                                index=X_test.index)

    # Train the OLS model
    ols_model = sm.OLS(y_train, X_train_const).fit()

    # Predict on training and testing sets
    y_train_pred = ols_model.predict(X_train_const)
    y_test_pred = ols_model.predict(X_test_const)

    # Evaluate the model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Print the summary with feature names
    print(ols_model.summary())
    print(f"\nTraining R^2: {train_r2:.3f}")
    print(f"Test R^2: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")

    return ols_model, train_r2, test_r2, test_rmse

def kfold_train_and_test_ols_with_plot(data, train_features, target_features, k=5, random_state=42):
    """
    Train and test a linear model using OLS with K-Fold Cross-Validation and plot the performance metrics.

    Parameters:
        data (pd.DataFrame): DataFrame containing the features and target.
        train_features (list of str): List of feature columns used for training.
        target_features (list of str): Target column name (list with one element).
        k (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Average R^2 and RMSE scores for training and testing sets.
    """
    # Extract features and target
    X = data[train_features]
    y = data[target_features[0]]

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Lists to store metrics for each fold
    train_r2_scores = []
    test_r2_scores = []
    test_rmse_scores = []
    fold_indices = list(range(1, k + 1))

    # Loop through each fold
    for train_index, test_index in kf.split(X):
        # Split data into train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Normalize the features (standardization: mean=0, std=1)
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        # Add a constant for the intercept term
        X_train_const = pd.DataFrame(sm.add_constant(X_train_normalized, has_constant="add"),
                                      columns=["const"] + train_features,
                                      index=X_train.index)
        X_test_const = pd.DataFrame(sm.add_constant(X_test_normalized, has_constant="add"),
                                     columns=["const"] + train_features,
                                     index=X_test.index)

        # Train the OLS model
        ols_model = sm.OLS(y_train, X_train_const).fit()

        # Predict on training and testing sets
        y_train_pred = ols_model.predict(X_train_const)
        y_test_pred = ols_model.predict(X_test_const)

        # Evaluate the model
        train_r2_scores.append(r2_score(y_train, y_train_pred))
        test_r2_scores.append(r2_score(y_test, y_test_pred))
        test_rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    # Calculate average metrics across all folds
    avg_train_r2 = np.mean(train_r2_scores)
    avg_test_r2 = np.mean(test_r2_scores)
    avg_test_rmse = np.mean(test_rmse_scores)

    # Plot R² and RMSE metrics for each fold
    plt.figure(figsize=(12, 6))

    # R² Plot
    plt.subplot(1, 2, 1)
    plt.plot(fold_indices, train_r2_scores, marker='o', label="Train R²")
    plt.plot(fold_indices, test_r2_scores, marker='o', label="Test R²")
    plt.axhline(y=avg_test_r2, color='r', linestyle='--', label="Avg Test R²")
    plt.title("R² Scores Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.xticks(fold_indices)
    plt.legend()

    # RMSE Plot
    plt.subplot(1, 2, 2)
    plt.plot(fold_indices, test_rmse_scores, marker='o', label="Test RMSE", color='g')
    plt.axhline(y=avg_test_rmse, color='r', linestyle='--', label="Avg Test RMSE")
    plt.title("RMSE Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.xticks(fold_indices)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nAverage Training R^2: {avg_train_r2:.3f}")
    print(f"Average Test R^2: {avg_test_r2:.3f}")
    print(f"Average Test RMSE: {avg_test_rmse:.3f}")

    return {
        "avg_train_r2": avg_train_r2,
        "avg_test_r2": avg_test_r2,
        "avg_test_rmse": avg_test_rmse
    }



def kfold_ridge_validation_with_plot(data, train_features, target_feature, n_splits=5, random_state=42, alpha=1.0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X = data[train_features]
    y = data[target_feature]

    train_r2_scores = []
    test_r2_scores = []
    test_rmse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train Ridge Regression
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train, y_train)

        # Predict
        y_train_pred = ridge_model.predict(X_train)
        y_test_pred = ridge_model.predict(X_test)

        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        test_rmse_scores.append(test_rmse)

    # Plot Results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_splits + 1), train_r2_scores, label="Train R²", marker="o")
    plt.plot(range(1, n_splits + 1), test_r2_scores, label="Test R²", marker="o")
    plt.xlabel("Fold Number")
    plt.ylabel("R² Score")
    plt.title("K-Fold Validation (Ridge): R² Scores")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_splits + 1), test_rmse_scores, label="Test RMSE", marker="o", color="red")
    plt.xlabel("Fold Number")
    plt.ylabel("RMSE")
    plt.title("K-Fold Validation (Ridge): Test RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Mean Training R²: {np.mean(train_r2_scores):.3f}")
    print(f"Mean Test R²: {np.mean(test_r2_scores):.3f}")
    print(f"Mean Test RMSE: {np.mean(test_rmse_scores):.3f}")

    return np.mean(train_r2_scores), np.mean(test_r2_scores), np.mean(test_rmse_scores)



def create_polygon(lat, long, crs_epsg=27700, buffer=500):
    """
    Create a 500 m x 500m polygon centered at (lat, long).

    Args:
        lat (float): Latitude of the center.
        long (float): Longitude of the center.
        crs_epsg (int): EPSG code for the CRS to use for the polygon.
        buffer (int): Buffer distance in meters (default is 1000 meters).

    Returns:
        GeoDataFrame: A GeoDataFrame containing the 1 km x 1 km polygon.
    """
    # Convert lat/long to a projected CRS for accurate measurements
    point = gpd.GeoDataFrame(
        {'geometry': [Point(long, lat)]},
        crs="EPSG:4326"
    ).to_crs(epsg=crs_epsg)

    # Extract x and y coordinates
    x, y = point.geometry.x[0], point.geometry.y[0]

    # Create a bounding box
    polygon = box(x - buffer / 2, y - buffer / 2, x + buffer / 2, y + buffer / 2)

    # Convert to GeoDataFrame
    ns_sec_gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=f"EPSG:{crs_epsg}")
    ns_sec_gdf['OA21CD'] = ['temp'] 

    return ns_sec_gdf

def predict_census(lat, long, model, osm_gdf, features_to_keep, feature_buffer_distances, crs_epsg=27700):
    """
    Predict metric for a given latitude and longitude.

    Args:
        lat (float): Latitude of the center of the polygon.
        long (float): Longitude of the center of the polygon.
        model (sm.OLS): Trained OLS model.
        osm_gdf (GeoDataFrame): GeoDataFrame containing features.
        features_to_keep (list): List of feature columns to use for prediction.
        crs_epsg (int): EPSG code for the CRS (default is EPSG:27700).

    Returns:
        float: Predicted ns_sec_l15 value.
    """
    # Create a polygon around the coordinates
    census_gdf = create_polygon(lat, long, crs_epsg, buffer= 500)

    # Count features within the polygon using the provided function
    result = assess.count_features_within_polygons(census_gdf, osm_gdf, features_to_keep, feature_buffer_distances,crs_epsg=crs_epsg)


    # Ensure all `features_to_keep` are present
    for feature in features_to_keep:
        if feature not in result.columns:
            result[feature] = 0  # Add missing feature with value 0

    # Extract the features and prepare for prediction
    feature_vector = result[features_to_keep].iloc[0].fillna(0).values.reshape(1, -1)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(feature_vector)

    # Add constant for OLS model
    feature_df = pd.DataFrame(X_normalized, columns=features_to_keep)
    feature_df = sm.add_constant(feature_df, has_constant="add")

    # Predict using the trained model
    predicted_value = model.predict(feature_df).values[0]

    return predicted_value





def simulate_future_features(data, old_features, year_suffix="2021"):
    """
    Replace old features (e.g., 2011) with updated features (e.g., 2021).
    """
    data_copy = data.copy()
    for feature in old_features:
        data_copy[feature] = data_copy[feature.replace("2011", year_suffix)]
    return data_copy

def simulate_infrastructure_change(data, feature_name, increment=1):
    """
    Simulate an increase in infrastructure (e.g., railway station count).
    """
    data[feature_name] += increment
    return data

def normalize_features(data, features, scaler=None):
    """
    Standardize features using StandardScaler (mean=0, std=1).
    """
    if scaler is None:
        scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[features])
    return pd.DataFrame(data_normalized, columns=features, index=data.index), scaler

def predict_future_car_usage(data, model, train_features, scaler):
    """
    Predict future car usage using a trained model with normalized features.
    """
    # Normalize the features using the pre-fitted scaler
    X_normalized = scaler.transform(data[train_features])
    X_const = sm.add_constant(X_normalized, has_constant="add")
    return model.predict(X_const)

def calculate_car_usage_change(data, future_column, baseline_column, output_column):
    """
    Calculate change in car usage between future and baseline years.
    """
    data[output_column] = data[future_column] - data[baseline_column]
    return data

def rank_and_prioritize(data, sort_column, columns_to_display, ascending=True):
    """
    Rank and prioritize regions based on a specified column.
    """
    ranked_data = data.sort_values(by=sort_column, ascending=ascending)
    return ranked_data[columns_to_display]

# Main Function
def prioritize_counties_for_infrastructure(data, old_features, change_features, target_model, increment=1):
    """
    Simulate the impact of public transport infrastructure changes and prioritize counties.
    """
    # Step 1: Simulate future features (replace 2011 with 2021)
    data_simulated = simulate_future_features(data, old_features, year_suffix="2021")

    # Step 2: Simulate infrastructure change
    data_simulated = simulate_infrastructure_change(data_simulated, "railway_station_count_change", increment)

    # Step 3: Normalize features using the same scaling process
    train_features = change_features + old_features
    X_normalized, scaler = normalize_features(data, train_features)

    # Ensure the simulated data uses the same normalization scaler
    data_simulated_normalized = pd.DataFrame(scaler.transform(data_simulated[train_features]), 
                                             columns=train_features, 
                                             index=data_simulated.index)

    # Add a constant for the intercept term
    X_const = sm.add_constant(data_simulated_normalized, has_constant="add")

    # Step 4: Predict future car usage
    data_simulated["car_usage_2031"] = target_model.predict(X_const)

    # Step 5: Calculate change in car usage
    data_simulated = calculate_car_usage_change(data_simulated, 
                                                future_column="car_usage_2031",
                                                baseline_column="car_usage_2021",
                                                output_column="car_usage_change_2031")

    # Step 6: Rank counties by car usage reduction
    columns_to_display = ["UTLA22NM", "car_usage_change_2031", "geometry"]
    top_priority_counties = rank_and_prioritize(data_simulated, 
                                                sort_column="car_usage_change_2031",
                                                columns_to_display=columns_to_display,
                                                ascending=True)

    return top_priority_counties

def create_design_matrices_uk(age_groups, split_point=55):
    ages_before = age_groups[age_groups <= split_point]
    ages_after = age_groups[age_groups > split_point]

    # Sinusoidal basis for ages ≤ split_point
    sin_basis_before = np.column_stack([
        np.sin(ages_before / 10),
        np.sin(ages_before / 10)**2,
        ages_before
    ])

    # Quadratic basis for ages > split_point
    poly_basis_after = np.column_stack([
        ages_after,
        ages_after**2
    ])

    # Add intercepts
    design_matrix_before = np.hstack((np.ones((len(ages_before), 1)), sin_basis_before))
    design_matrix_after = np.hstack((np.ones((len(ages_after), 1)), poly_basis_after))

    return design_matrix_before, design_matrix_after

def create_design_matrices_camb(age_groups, split_point=18):
    ages_before = age_groups[age_groups <= split_point]
    ages_after = age_groups[age_groups > split_point]

    # linear basis for ages ≤ split_point
    basis_before = np.column_stack([
        ages_before
    ])

    # Quadratic basis for ages > split_point
    poly_basis_after = np.column_stack([
        ages_after,
        ages_after**2
    ])

    # Add intercepts
    design_matrix_before = np.hstack((np.ones((len(ages_before), 1)), basis_before))
    design_matrix_after = np.hstack((np.ones((len(ages_after), 1)), poly_basis_after))

    return design_matrix_before, design_matrix_after


def fit_and_predict(population, design_matrix_before, design_matrix_after, split_point):
    model_before = sm.OLS(population[:split_point + 1], design_matrix_before)
    results_before = model_before.fit()
    predictions_before = results_before.predict(design_matrix_before)

    model_after = sm.OLS(population[split_point + 1:], design_matrix_after)
    results_after = model_after.fit()
    predictions_after = results_after.predict(design_matrix_after)

    return predictions_before, predictions_after

def fit_and_predict_with_ci(population, design_matrix_before, design_matrix_after, split_point):
    model_before = sm.OLS(population[:split_point + 1], design_matrix_before)
    results_before = model_before.fit()
    predictions_before = results_before.predict(design_matrix_before)
    conf_int_before = results_before.get_prediction(design_matrix_before).conf_int(alpha=0.05)

    model_after = sm.OLS(population[split_point + 1:], design_matrix_after)
    results_after = model_after.fit()
    predictions_after = results_after.predict(design_matrix_after)
    conf_int_after = results_after.get_prediction(design_matrix_after).conf_int(alpha=0.05)

    return predictions_before, predictions_after, conf_int_before, conf_int_after

def predict_and_analyse_r2(merged_df, feature_cols, all_cols=False):
    X = merged_df[feature_cols]
    y = merged_df[21].values

    if(not all_cols):
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(model.summary())

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    print("R^2 Score:", r2)

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.7, color="blue", edgecolor="black")
    plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Perfect Correlation")
    plt.xlabel("Actual Percentage of 21-Year-Olds")
    plt.ylabel("Predicted Percentage of 21-Year-Olds")
    plt.title("Correlation Between Actual and Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def k_fold_cross_validation_ols(merged_df, feature_cols, target_col, k_values):
    X = merged_df[feature_cols].values
    y = merged_df[target_col].values
    
    if(len(feature_cols)!=9):
      X = sm.add_constant(X)
    
    mean_test_r2_scores = []
    mean_train_r2_scores = []
    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        test_r2_scores = []
        train_r2_scores = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = sm.OLS(y_train, X_train).fit()
            
            y_test_pred = model.predict(X_test)
            test_r2_scores.append(r2_score(y_test, y_test_pred))
            
            y_train_pred = model.predict(X_train)
            train_r2_scores.append(r2_score(y_train, y_train_pred))

            # print("The root mean squared error for k=" + str(k) + " is " + str(mean_squared_error(y_test, y_test_pred)))
        
        mean_test_r2_scores.append(np.mean(test_r2_scores))
        mean_train_r2_scores.append(np.mean(train_r2_scores))
        
        print(f"k={k}, Mean Test R^2: {np.mean(test_r2_scores):.4f}, Mean Train R^2: {np.mean(train_r2_scores):.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, mean_test_r2_scores, marker='o', linestyle='-', color='b', label="Mean Test R^2")
    plt.plot(k_values, mean_train_r2_scores, marker='o', linestyle='-', color='r', label="Mean Train R^2")
    plt.title("Cross-Validation Performance vs. k (OLS)")
    plt.xlabel("Number of Folds (k)")
    plt.ylabel("Mean R^2 Score")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mean_test_r2_scores, mean_train_r2_scores


def k_fold_cross_validation_regularised_sm(merged_df, feature_cols, target_col, k_values, alpha_values, l1_weights):
    X = merged_df[feature_cols].div(merged_df["total"], axis=0).values
    y = merged_df[target_col].values

    if(len(feature_cols)!=9):
      X = sm.add_constant(X)

    results = {}

    for alpha in alpha_values:
        for l1_wt in l1_weights:
            mean_test_r2_scores = []
            mean_train_r2_scores = []
            mean_baseline_scores = []

            for k in k_values:
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                test_r2_scores = []
                train_r2_scores = []
                baseline_scores = []

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = sm.OLS(y_train, X_train)
                    fitted_model = model.fit_regularized(alpha=alpha, L1_wt=l1_wt)
                    baseline_model = model.fit()

                    y_test_pred = fitted_model.predict(X_test)
                    test_r2_scores.append(r2_score(y_test, y_test_pred))

                    y_train_pred = fitted_model.predict(X_train)
                    train_r2_scores.append(r2_score(y_train, y_train_pred))

                    y_baseline_pred = baseline_model.predict(X_test)
                    baseline_scores.append(r2_score(y_test, y_baseline_pred))

                mean_test_r2_scores.append(np.mean(test_r2_scores))
                mean_train_r2_scores.append(np.mean(train_r2_scores))
                mean_baseline_scores.append(np.mean(baseline_scores))

            results[(alpha, l1_wt)] = {
                "mean_test_r2_scores": mean_test_r2_scores,
                "mean_train_r2_scores": mean_train_r2_scores,
                "mean_baseline_scores": mean_baseline_scores
            }

            print(f"Alpha={alpha}, L1_wt={l1_wt}: Mean Test R^2={np.mean(mean_test_r2_scores):.4f}, "
                  f"Mean Train R^2={np.mean(mean_train_r2_scores):.4f}", f"Mean Baseline R^2={np.mean(mean_baseline_scores):.4f}")
            plt.figure(figsize=(8, 5))
            plt.plot(k_values, mean_test_r2_scores, marker='o', linestyle='-', label=f"Mean Test R^2 (alpha={alpha}, L1_wt={l1_wt})")
            plt.plot(k_values, mean_train_r2_scores, marker='o', linestyle='--', label=f"Mean Train R^2 (alpha={alpha}, L1_wt={l1_wt})")
            plt.plot(k_values, mean_baseline_scores, marker='o', linestyle='-.', label=f"Mean Baseline R^2 (alpha={alpha}, L1_wt={l1_wt})")
            plt.title(f"Cross-Validation Performance vs. k (Alpha={alpha}, L1_wt={l1_wt})")
            plt.xlabel("Number of Folds (k)")
            plt.ylabel("Mean R^2 Score")
            plt.legend()
            plt.grid(True)
            plt.show()

    return results

def fit_and_plot_coefficients(merged_df, age_df, feature_cols, alpha=0.0001, l1_wt=0.0):
    X = merged_df[feature_cols].values

    if(len(feature_cols)!=9):
      X = sm.add_constant(X)

    coefficients = [[] for i in range(9)]

    for age in range(100):
        y =  merged_df[[age]]
        model = sm.OLS(y, X).fit()
        for i in range(9):
          coefficients[i] += [model.params[i]]


    fig = plt.figure(figsize=(10,8))
    for i in range(9):
        plt.plot(range(100), coefficients[i], label=feature_cols[i])
    plt.legend(fontsize=9)
    plt.show()
    

    return coefficients

def predict_and_plot_city_age_profile(merged_df,age_df, feature_cols, city_name, alpha=0.0001, l1_wt=0.0):
    X = merged_df[feature_cols].values
    if(len(feature_cols)!=9):
      X = sm.add_constant(X)

    predicted_age_profile = []

    for age in range(100):
        y = merged_df[[age]].values
        model = sm.OLS(y, X).fit_regularized(alpha=alpha, L1_wt=l1_wt)
        city_features = merged_df.loc[merged_df["geography"] == city_name, feature_cols].values
        y_pred = model.predict(city_features)
        predicted_age_profile.append(y_pred[0])

    actual_age_profile = age_df.loc[city_name]
    predicted_age_profile = predicted_age_profile / np.sum(predicted_age_profile) *  age_df.loc[city_name].sum(axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(100), actual_age_profile, label="Actual Age Profile", color="blue", linewidth=2)
    plt.plot(range(100), predicted_age_profile, label="Predicted Age Profile", color="red", linestyle="--", linewidth=2)
    plt.title(f"Age Profile Prediction for {city_name}")
    plt.xlabel("Age Group")
    plt.ylabel("Population Share")
    plt.legend(loc="upper right", fontsize="medium", frameon=False, title="Profile")
    plt.grid(True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    residuals = np.array(actual_age_profile) - np.array(predicted_age_profile)
    mse = np.mean(residuals ** 2)
    print(f"Mean Squared Error (MSE) for {city_name}: {mse:.5f}")

    return predicted_age_profile, mse