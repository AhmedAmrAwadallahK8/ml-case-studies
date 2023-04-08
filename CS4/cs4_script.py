import sys
import os
from scipy.io import arff
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn.metrics as mt
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score


class UnexpectedFileStructureError(Exception):
    def __init__(self):
        msg = "This script expects a file path to a directory that contains "
        msg += "CS4 Bankruptcy Dataset. The Bankruptcy dataset is a "
        msg += "group of five arff files with the names, 1year, 2year, "
        msg += "3year, 4year, 5year. Did you specify a folder path that "
        msg += "contains these files?"
        super().__init__(msg)


class MissingInputError(Exception):
    def __init__(self):
        msg = "This script expects exactly one input, a file directory that "
        msg += "contains the bankruptcy dataset. You input nothing. "
        msg += "Did you forget to add an argument with the script?"
        super().__init__(msg)


class ExcessiveInputError(Exception):
    def __init__(self):
        msg = "This script expects exactly one input, a file directory that "
        msg += "contains the bankruptcy dataset. You called this script "
        msg += "with more than one argument. Only use one and try again. "
        super().__init__(msg)


def check_correct_file_structure(inputs):
    data_direc = inputs[1]
    expected_files = ["1year.arff",
                      "2year.arff",
                      "3year.arff",
                      "4year.arff",
                      "5year.arff"]
    for root, dirs, files in os.walk(data_direc):
        for f in files:
            if f not in expected_files:
                print(f)
                raise UnexpectedFileStructureError


def check_valid_inputs(inputs):
    if len(inputs) < 2:
        raise MissingInputError
    elif len(inputs) > 2:
        raise ExcessiveInputError


def load_data(inputs):
    data_direc = inputs[1]
    raw_year_data = []
    for root, dirs, files in os.walk(data_direc):
        for f in files:
            file_path = os.path.join(root, f)
            arff_data = arff.loadarff(file_path)
            raw_year_data.append(pd.DataFrame(arff_data[0]))
    return raw_year_data


def log_section(title="No Title", content="No Content"):
    log_file.write(title + "\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    log_file.write(content + "\n")
    log_file.write("\n")


def preprocess_data(raw_df):
    # Seperate features by percentage of data NA
    na_variations = [np.nan]
    below_50_precent_na_features = []  # Mean Impute
    above_50_perecent_na_features = []  # Remove

    for feature in list(raw_df):
        percent_na = raw_df[feature].isin(na_variations).sum()
        percent_na = percent_na / (raw_df.shape[0]) * 100
        percent_na = round(percent_na, 3)
        if percent_na <= 50:
            below_50_precent_na_features.append(feature)
        if percent_na > 50:
            above_50_perecent_na_features.append(feature)

    # Handle NAs
    clean_df = raw_df.copy()
    # Mean Inpute if less than 50% missing
    mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    clean_df[below_50_precent_na_features] = mean_imp.fit_transform(
        clean_df[below_50_precent_na_features])

    # Remove if more than 50% missing
    for feature in above_50_perecent_na_features:
        clean_df = clean_df.drop([feature], axis=1)

    # Changing Target Feature to be int type
    target_feature = ["class"]
    cts_features = []
    for feature in list(clean_df):
        if feature not in target_feature:
            cts_features.append(feature)
    clean_df[target_feature] = clean_df[target_feature].astype(int)

    return clean_df


if __name__ == '__main__':
    inputs = sys.argv
    check_valid_inputs(inputs)
    check_correct_file_structure(inputs)
    print("Full Execution on my machine takes around 20 Minutes")
    # Log Start
    log_file = open("log.txt", "w")
    log_file.write("Case Study 4 Report\n\n")

    # Use input to load data
    raw_years = load_data(inputs)

    # Group all years into one large dataset and shuffle
    raw_df = pd.concat(raw_years, axis=0, ignore_index=True)

    # Data Preprocessing
    clean_df = preprocess_data(raw_df)

    # Target Feature Mappings
    bankrupt_to_id_map = {"Not Bankrupt": 0, "Bankrupt": 1}
    id_to_bankrupt_map = {0: "Not Bankrupt", 1: "Bankrupt"}

    # Shuffle
    clean_df = clean_df.sample(frac=1)

    # Seperate Features and Target
    target_feature = ["class"]
    y = clean_df[target_feature[0]]
    x = clean_df.drop(target_feature, axis=1)

    ninety_precent_of_data = int(x.shape[0]*0.9)
    ten_precent_of_data = int(x.shape[0]*0.1)
    train_x = x.head(ninety_precent_of_data)
    train_y = y.head(ninety_precent_of_data)
    val_x = x.tail(ten_precent_of_data)
    val_y = y.tail(ten_precent_of_data)

    # Random Forest Model
    rf_params = {'min_weight_fraction_leaf': 3.359818286283781e-06,
                 'min_samples_split': 10,
                 'min_samples_leaf': 6,
                 'min_impurity_decrease': 2.06913808111479e-05,
                 'max_depth': 429,
                 'criterion': 'entropy',
                 'n_jobs': 4,
                 'n_estimators': 1000}

    # Cross Val Score
    rf = RandomForestClassifier(**rf_params)
    splits = KFold(n_splits=10, shuffle=True)
    cross_score_rf = cross_val_score(rf, x, y, cv=splits, scoring='roc_auc')
    auc_rf = cross_score_rf.mean()
    auc_rf_std = cross_score_rf.std()
    rf_cross_val_rep = f"auc={auc_rf:4f}, std={auc_rf_std:4f}"
    log_section(title="RF Cross Val AUC",
                content=rf_cross_val_rep)

    # Final Model For Analysis
    rf.fit(train_x, train_y)

    # XGB Model
    xg_params = {
        "objective": "binary:logistic",
        "nthread": 4,
        "eval_metric": "auc",
        "early_stopping_rounds": 5,
        "n_estimators": 200,
        'eta': 0.6309573444801936,
        'max_depth': 12,
        'min_child_weight': 0.00630957344480193,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'lambda': 10}

    # Cross Val Score
    cross_score_xgb = []
    split_count = 10
    kf = KFold(n_splits=split_count)
    for id, (train_i, test_i) in enumerate(kf.split(x, y)):
        x_train = x.iloc[train_i]
        y_train = y.iloc[train_i]
        x_test = x.iloc[test_i]
        y_test = y.iloc[test_i]
        xg_model = XGBClassifier(**xg_params)
        xg_model.fit(x_train,
                     y_train,
                     eval_set=[(x_test, y_test)],
                     verbose=False)
        y_pred = xg_model.predict_proba(x_test)[:, 1]
        cross_score_xgb.append(roc_auc_score(y_test, y_pred))

    cross_score_xgb = np.asarray(cross_score_xgb)
    auc_xgb = cross_score_xgb.mean()
    auc_xgb_std = cross_score_xgb.std()
    xgb_cross_val_rep = f"auc={auc_xgb:4f}, std={auc_xgb_std:4f}"
    log_section(title="XGB Cross Val AUC",
                content=xgb_cross_val_rep)

    # Final Model For Analysis
    xg_params_no_early = {
        "objective": "binary:logistic",
        "nthread": 4,
        "n_estimators": 200,
        'eta': 0.6309573444801936,
        'max_depth': 12,
        'min_child_weight': 0.00630957344480193,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'lambda': 10}
    xg_model = XGBClassifier(**xg_params_no_early)
    xg_model.fit(train_x, train_y, verbose=False)

    # Random Forest Analysis
    # Classification Report
    y_pred_final = rf.predict(val_x)
    y_true = val_y
    classif_rep = classification_report(
        y_true,
        y_pred_final
    )
    log_section(title="Random Forest Classification Report",
                content=classif_rep)

    # AUC and ROC Curve All Data
    y_pred_prob = rf.predict_proba(val_x)
    pos_class = 1
    y_label = id_to_bankrupt_map[pos_class]
    preds = y_pred_prob[:, pos_class]
    fpr_tree, tpr_tree, thresholds = mt.roc_curve(val_y,
                                                  preds,
                                                  pos_label=pos_class)
    auc_report = f"{y_label} AUC: {mt.auc(fpr_tree, tpr_tree):.4f}\n"
    log_section(title="Random Forest AUC Report",
                content=auc_report)

    # Feature Importance
    importances_rf = pd.DataFrame(
        rf.feature_importances_,
        index=x.columns,
        columns=["importance"]).sort_values("importance", ascending=False)
    fig, ax = plt.subplots()
    importances_rf.head(5).plot.barh(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.savefig("rf_feature_importance.png", bbox_inches='tight')
    plt.clf()

    # XGB Analysis
    # Classification Report
    y_pred_final = xg_model.predict(val_x)
    y_true = val_y
    classif_rep = classification_report(
        y_true,
        y_pred_final
    )
    log_section(title="XGB Classification Report",
                content=classif_rep)

    # AUC and ROC Curve All Data
    y_pred_prob = xg_model.predict_proba(val_x)
    pos_class = 1
    y_label = id_to_bankrupt_map[pos_class]
    preds = y_pred_prob[:, pos_class]
    fpr_xgb, tpr_xgb, thresholds = mt.roc_curve(val_y,
                                                preds,
                                                pos_label=pos_class)
    auc_report = f"{y_label} AUC: {mt.auc(fpr_xgb, tpr_xgb):.4f}\n"
    log_section(title="XGB AUC Report",
                content=auc_report)

    # Feature Importance
    plot_importance(xg_model, max_num_features=5)
    plt.savefig("xgb_feature_importance.png", bbox_inches='tight')
    plt.clf()

    # ROC Comparison Plot
    plt.plot(fpr_tree, tpr_tree, label="Random Forest ROC")
    plt.plot(fpr_xgb, tpr_xgb, label="XGB ROC")
    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig("roc_plot.png", bbox_inches='tight')
    plt.clf()

    log_file.write("\n")
    log_file.close()
