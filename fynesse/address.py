# This file contains code for suporting addressing questions in the data
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


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