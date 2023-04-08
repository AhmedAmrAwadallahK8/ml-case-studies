import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Your case study is to build a linear regression model using L1 or L2
# regularization (or both) the task to predict the Critical Temperature as
# closely as possible. In addition, include in your write-up which variable
# carries the most importance.

INVALID_INPUT = 1


def try_to_read_input_csvs(inputs):
    try:
        read_input_csvs(inputs)
    except Exception:
        print("This script expects the two csv files associated with Case "
              "Study1. You either input nothing, 1 file path, or one or more "
              "of your inputs contains an invalid path.")
        sys.exit(INVALID_INPUT)


def read_input_csvs(inputs):
    global df1, df2
    df1_path = inputs[1]
    df2_path = inputs[2]
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)


def linear_cv_feature_selection(x, y, features, print_progress=False):
    best_alpha_lasso = 0
    best_rmse_lasso = 0
    first = True
    report_freq = 1
    iter_count = 5
    param_alpha = 0.01
    for i in range(0, iter_count):
        param_alpha = param_alpha*10

        lasso_model = Lasso(alpha=param_alpha)

        rmse_lasso_raw = cross_val_score(lasso_model, x, y, cv=5,
                                         scoring='neg_root_mean_squared_error')

        rmse_lasso = -1*sum(rmse_lasso_raw)/len(rmse_lasso_raw)

        if first:
            first = False
            best_alpha_lasso = param_alpha
            best_rmse_lasso = rmse_lasso
        else:
            if (rmse_lasso < best_rmse_lasso):
                best_alpha_lasso = param_alpha
                best_rmse_lasso = rmse_lasso

        if ((i+1) % report_freq == 0) and print_progress:
            print(f"Iteration {i+1}. "
                  f"Percent done = {(i+1)/iter_count*100:.4f}%")

    # print(f"Best lasso rmse={best_rmse_lasso:.4f} alpha={best_alpha_lasso}")
    lasso_model = Lasso(alpha=best_alpha_lasso)
    lasso_model.fit(x, y)
    nonzero_features = list(lasso_model.coef_ > 0)
    selected_features = []

    for i in range(len(nonzero_features)):
        if nonzero_features[i]:
            selected_features.append(features[i])
    return selected_features


def linear_cv_compare(x, y, print_progress=False):
    best_alpha_ridge = 0
    best_rmse_ridge = 0
    first = True
    report_freq = 10
    iter_count = 100
    i = 0
    t1 = time.time()
    for param_alpha in np.logspace(-6, 6, iter_count):
        i += 1

        ridge_model = Ridge(alpha=param_alpha)

        rmse_ridge_raw = cross_val_score(ridge_model, x, y, cv=5,
                                         scoring='neg_root_mean_squared_error')

        rmse_ridge = -1*sum(rmse_ridge_raw)/len(rmse_ridge_raw)

        if first:
            first = False
            best_alpha_ridge = param_alpha
            best_rmse_ridge = rmse_ridge
        else:
            if (rmse_ridge < best_rmse_ridge):
                best_alpha_ridge = param_alpha
                best_rmse_ridge = rmse_ridge

        if ((i+1) % report_freq == 0) and print_progress:
            print(f"Iteration {i+1}. "
                  f"Percent done = {(i+1)/iter_count*100:.4f}%")
    t2 = time.time()
    elapsed_time = t2-t1

    print(f"Best ridge rmse={best_rmse_ridge:.4f}, "
          f"alpha={best_alpha_ridge}, time={elapsed_time:.4f}s")

    return best_alpha_ridge


def get_top_n_coef(feature_list, coef_list, n=5):
    feature_scores_abs = {}
    feature_scores = {}

    for i in range(len(feature_list)):
        feature = feature_list[i]
        coef = coef_list[i]
        feature_scores_abs[feature] = abs(coef)
        feature_scores[feature] = coef

    feature_scores_abs = dict(sorted(feature_scores_abs.items(),
                              key=lambda item: item[1], reverse=True))
    top_n_features = list(feature_scores_abs.keys())[0:n]

    top_feature_scores = {}
    for feature in top_n_features:
        top_feature_scores[feature] = feature_scores[feature]

    return top_feature_scores


if __name__ == '__main__':
    inputs = sys.argv

    try_to_read_input_csvs(inputs)

    # Combine dataframes
    y = df1["critical_temp"]
    df1 = df1.drop(["critical_temp"], axis=1)
    df2 = df2.drop(["critical_temp", "material"], axis=1)

    # Find which dataframe contains the cts variables that need to be scaled
    df1_features = list(df1.columns)
    df2_features = list(df2.columns)
    if "number_of_elements" in df1_features:
        cts_features = df1_features
    else:
        cts_features = df2_features

    # Scale cts data
    scaler = StandardScaler()
    train = pd.concat([df1, df2], axis=1)
    train[cts_features] = scaler.fit_transform(train[cts_features])

    # Lasso Feature Selection
    print("Lasso Feature Selection Starting. "
          "This step takes a little bit of time...")
    selected_features_main = linear_cv_feature_selection(train,
                                                         y,
                                                         list(train.columns))
    train_fs = train[selected_features_main]
    print("33% Done")

    # Ridge Results
    print("---All Data---")
    best_alpha_all_data = linear_cv_compare(train, y)
    all_data_model = Ridge(alpha=best_alpha_all_data)
    all_data_model.fit(train, y)
    all_data_top5 = get_top_n_coef(train.columns, all_data_model.coef_)
    print("Top 5 Features and Coefficients")
    for feature in all_data_top5.keys():
        coef = all_data_top5[feature]
        print(f"{feature}: {coef:.4f}")

    print("66% Done")

    print("---Feature Selected Data---")
    best_alpha_fs = linear_cv_compare(train_fs, y)
    feature_selected_model = Ridge(alpha=best_alpha_fs)
    feature_selected_model.fit(train_fs, y)
    feature_selected_top5 = get_top_n_coef(train_fs.columns,
                                           feature_selected_model.coef_)
    for feature in feature_selected_top5.keys():
        coef = feature_selected_top5[feature]
        print(f"{feature}: {coef:.4f}")

    print("100% Done")

    # Y_pred to True plots
    figure, axis = plt.subplots(1, 2, figsize=(16, 10))

    y_pred_all_data = all_data_model.predict(train)
    axis[0].scatter(y_pred_all_data, y, label="Y Pred v Y True")
    axis[0].scatter(y, y, label="Example of desired result")
    axis[0].set_title("All Data Ypred to Ytrue comparison")
    axis[0].set_xlabel("Y Predicted")
    axis[0].set_ylabel("Y True")
    axis[0].legend(loc='upper center', shadow=True)

    y_pred_fs = feature_selected_model.predict(train_fs)
    axis[1].scatter(y_pred_fs, y, label="Y Pred v Y True")
    axis[1].scatter(y, y, label="Example of desired result")
    axis[1].set_title("Feature Selected Ypred to Ytrue comparison")
    axis[1].set_xlabel("Y Predicted")
    axis[1].set_ylabel("Y True")
    axis[1].legend(loc='upper center', shadow=True)

    plt.show()

    print("Program Finished")
