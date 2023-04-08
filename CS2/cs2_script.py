import pandas as pd
import numpy as np
import math
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics as mt

# Your case study is to build a classifier using logistic regression to
# predict hospital readmittance. There is missing data that must be imputed.
# Once again, discuss variable importances as part of your submission.


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


def get_mapping(range, id_map_df):
    map_dict = {}
    for i in range:
        id = int(id_map_df.loc[i][0])
        descrip = id_map_df.loc[i][1]
        if type(descrip) == float and math.isnan(descrip):
            descrip = None
        map_dict[id] = descrip
    return map_dict


def logreg_cv_compare(x, y, print_progress=False, n_jobs=1):
    best_C = 0
    best_f1 = 0
    first = True
    iter_count = 20
    report_freq = int(iter_count/20)
    i = 0
    t1 = time.time()
    for param_C in np.logspace(-6, 0, iter_count):

        model = LogisticRegression(multi_class='multinomial',
                                   solver='lbfgs',
                                   C=param_C,
                                   n_jobs=6,
                                   max_iter=1000)

        f1_cv = cross_val_score(model, x, y, cv=5,
                                scoring='f1_macro')

        f1 = sum(f1_cv)/len(f1_cv)

        if first:
            first = False
            best_C = param_C
            best_f1 = f1
        else:
            if (f1 > best_f1):
                best_C = param_C
                best_f1 = f1

        if ((i+1) % report_freq == 0) and print_progress:
            print(f"Iteration {i+1}. "
                  f"Percent done = {(i+1)/iter_count*100:.4f}%")

        i += 1
    t2 = time.time()
    elapsed_time = t2-t1

    report_str = f"Best f1={best_f1:.4f}, "
    report_str += f"C={best_C}, time={elapsed_time:.4f}s\n"

    return best_C, report_str


def get_top_n_coef_multiclass_logreg(feature_list, coef_list, n=5):
    feature_scores_abs = {}
    top_n_per_class = []
    feature_scores = {}
    class_count = coef_list.shape[0]
    for class_id in range(class_count):
        for i in range(len(feature_list)):
            feature = feature_list[i]
            coef = coef_list[class_id, i]
            feature_scores_abs[feature] = abs(coef)
            feature_scores[feature] = coef

        feature_scores_abs = dict(sorted(feature_scores_abs.items(),
                                  key=lambda item: item[1], reverse=True))
        top_n_features = list(feature_scores_abs.keys())[0:n]

        top_feature_scores = {}
        for feature in top_n_features:
            top_feature_scores[feature] = feature_scores[feature]

        top_n_per_class.append(top_feature_scores)

    return top_n_per_class


def print_top_coefs_multiclass_logreg(top_n_coefs, id_targ_map):
    str_report = ""
    for class_id in range(len(top_n_coefs)):
        feature_top_n = top_n_coefs[class_id]
        class_name = id_targ_map[class_id]
        for feature in feature_top_n.keys():
            coef = feature_top_n[feature]
            str_report += f"Class: {class_name} | {feature}: {coef:.4f}\n"
        str_report += "--------------------------------\n"
    return str_report


def plot_heatmap(data,
                 xlab,
                 ylab,
                 size=9,
                 title="untitled heatmap",
                 cbar_label="untitled",
                 save_name="untitled.png"):

    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(data, cmap=plt.cm.Blues)

    for i in range(len(ylab)):
        for j in range(len(xlab)):
            ax.text(j, i, data[i, j],
                    ha="center", va="center", color="black")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(xlab)), labels=xlab)
    ax.set_yticks(np.arange(len(ylab)), labels=ylab)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    log_file = open("log.txt", "w")
    log_file.write("Case Study 2 Report\n\n")

    inputs = sys.argv
    try_to_read_input_csvs(inputs)

    # Find which dataframe contains the cts variables that need to be scaled
    if (df1.shape == (101766, 50)) and (df2.shape == (67, 2)):
        raw_data = df1
        id_map_df = df2
    elif (df2.shape == (101766, 50)) and (df1.shape == (67, 2)):
        raw_data = df2
        id_map_df = df1
    else:
        raise Exception("One or both of dataframes has unexpected shape")

    # Shuffle raw data
    raw_data = raw_data.sample(frac=1)

    # Use metadata to provide useful descriptions for some features
    admission_type_map = get_mapping(range(0, 8), id_map_df)
    discharge_map = get_mapping(range(10, 40), id_map_df)
    admission_source_map = get_mapping(range(42, 67), id_map_df)

    raw_data["admission_type_id"] = raw_data[
        "admission_type_id"].map(admission_type_map)
    raw_data["discharge_disposition_id"] = raw_data[
        "discharge_disposition_id"].map(discharge_map)
    raw_data["admission_source_id"] = raw_data[
        "admission_source_id"].map(admission_source_map)

    # Data Cleaning Decisions
    # Remove weight as a feature as too much of it is missing
    clean_df = raw_data.drop(["weight"], axis=1)
    # Remove the following features as they should have no relationship with
    # target
    features_to_remove = ["patient_nbr", "encounter_id"]
    clean_df = clean_df.drop(features_to_remove, axis=1)
    # Impute desired features
    clean_df["gender"] = clean_df["gender"].replace("Unknown/Invalid", "?")
    features_to_impute = ["race", "diag_1", "diag_2", "diag_3", "gender",
                          "admission_type_id", "admission_source_id",
                          "discharge_disposition_id"]
    # All these categories use "?" to signify na data so replace with None
    for feature in features_to_impute:
        clean_df[feature] = clean_df[feature].replace("?", None)
    imputer = SimpleImputer(missing_values=None, strategy="most_frequent")
    for feature in features_to_impute:
        imputed_feature = imputer.fit_transform(clean_df[[feature]]).ravel()
        clean_df[feature] = imputed_feature

    # Encode Target
    readmitted_to_id_map = {"NO": 0, "<30": 1, ">30": 2}
    id_to_readmitted_map = {0: "NO", 1: "<30", 2: ">30"}
    clean_df["readmitted"] = clean_df["readmitted"].map(readmitted_to_id_map)

    # Specifying cts v categorical features
    cts_features = ["time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                    "number_diagnoses"]

    categ_features = []
    for feature in list(clean_df):
        if feature not in cts_features:
            categ_features.append(feature)

    # Scaling Cts Data
    scaler = StandardScaler()
    clean_df[cts_features] = scaler.fit_transform(clean_df[cts_features])

    # Split data into train data and target data
    y = clean_df["readmitted"]
    x = clean_df.drop("readmitted", axis=1)

    #  One hot encode
    x = pd.get_dummies(x, drop_first=True)

    # Subset data for demonstration purposes
    subset_count = 500
    print("Data subsetting is ACTIVE, only", subset_count,
          "samples being used to train")
    print("This is just to demonstrate the code behavior quickly")
    x = x.head(subset_count)
    y = y.head(subset_count)

    # Search optimal regularization C
    print("Finding optimal model hyperparameters...")
    print("")
    optimal_c, report_str = logreg_cv_compare(
        x,
        y,
        print_progress=True,
        n_jobs=4)
    log_file.write("Optimization Report\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    log_file.write(report_str)
    log_file.write("\n")

    # Final Model
    final_model = LogisticRegression(multi_class='multinomial',
                                     solver='lbfgs',
                                     C=optimal_c,
                                     max_iter=1000,
                                     n_jobs=6)
    final_model.fit(x, y)
    y_pred = final_model.predict(x)

    # Final Model Analysis

    # Top Coefs Per Target Feature Class
    class_top_n = get_top_n_coef_multiclass_logreg(
        list(x),
        final_model.coef_,
        n=8)

    log_file.write("Most Relevant Coefs Report\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    log_file.write(
        print_top_coefs_multiclass_logreg(
            class_top_n,
            id_to_readmitted_map
        )
    )
    log_file.write("\n")

    # Print Race Coef
    log_file.write("Finding the Weights of the Race Category\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    races = ['race_Asian',
             'race_Caucasian',
             'race_Hispanic',
             'race_Other',
             'race_AfricanAmerican']
    class_race_score = []
    class_count = final_model.coef_.shape[0]
    for class_id in range(class_count):
        race_feature_score = {}

        for race in races:
            if race in list(x):
                race_ind = list(x).index(race)
                coef = final_model.coef_[class_id, race_ind]

                race_feature_score[race] = coef

        class_race_score.append(race_feature_score)
    log_file.write(
        print_top_coefs_multiclass_logreg(
            class_race_score,
            id_to_readmitted_map
        )
    )
    log_file.write("\n")

    # Classif Report
    log_file.write("Classification Report\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    log_file.write(
        classification_report(
            y,
            y_pred,
            target_names=id_to_readmitted_map.values()
        )
    )
    log_file.write("\n")

    # Conf Matrix Plot
    conf_mat = confusion_matrix(y, y_pred)

    base_labels = list(id_to_readmitted_map.values())
    label_true = []
    label_pred = []
    for i in range(len(base_labels)):
        true_label = base_labels[i] + " true"
        pred_label = base_labels[i] + " pred"
        label_pred.append(pred_label)
        label_true.append(true_label)

    plot_heatmap(conf_mat,
                 label_pred,
                 label_true,
                 size=8,
                 title="Confusion Matrix Heatmap",
                 cbar_label="Counts",
                 save_name="conf_heatmap.png")

    # ROC Curve and AUC
    log_file.write("ROC/AUC Metrics\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    y_pred_prob = final_model.predict_proba(x)
    for i in range(y_pred_prob.shape[1]):
        y_label = id_to_readmitted_map[i]
        preds = y_pred_prob[:, i]
        fpr, tpr, thresholds = mt.roc_curve(y, preds, pos_label=i)
        plt.plot(fpr, tpr, label=y_label)
        title_str = 'ROC Curve for multiclassifcation'
        plt.title(title_str)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        log_file.write(f"{y_label} AUC: {mt.auc(fpr, tpr):.4f}\n")

    plt.legend()
    plt.savefig("roc_plot.png", bbox_inches='tight')
    plt.clf()
    log_file.write("\n")
    log_file.close()
