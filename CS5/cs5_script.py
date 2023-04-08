
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


class UnexpectedFileStructureError(Exception):
    def __init__(self):
        msg = "This script expects a file path to a directory that contains "
        msg += "CS5 Network Dataset. The Network dataset is a "
        msg += "single csv labelled as log2.csv "
        msg += "Did you specify a folder path that "
        msg += "contains this file?"
        super().__init__(msg)


class MissingInputError(Exception):
    def __init__(self):
        msg = "This script expects atleast one input, a file directory that "
        msg += "contains the network dataset. You input nothing. "
        msg += "Did you forget to add an argument with the script?"
        super().__init__(msg)


class ExcessiveInputError(Exception):
    def __init__(self):
        msg = "This script expects atleast one input and at most 2."
        msg += " The first argument is the file directory containing the data"
        msg += " and the second optional one is a flag specifying if you want"
        msg += " to train on all the data or a subset"
        msg += " You called this script with more than two arguments."
        msg += " Only use one or two and try again."
        super().__init__(msg)


class UnexpectedFlagError(Exception):
    def __init__(self):
        msg = "Flag input can only have a value of 0 or 1. Script received"
        msg += " something unexpected instead."
        super().__init__(msg)


class NonexistantFolderError(Exception):
    def __init__(self):
        msg = "Input folder does not exist"
        super().__init__(msg)


def check_correct_file_structure(inputs):
    data_direc = inputs[1]
    expected_files = ["log2.csv"]
    if os.path.exists(data_direc):
        for root, dirs, files in os.walk(data_direc):
            if files == []:
                raise UnexpectedFileStructureError
            for f in files:
                if f not in expected_files:
                    print(f)
                    raise UnexpectedFileStructureError
    else:
        raise NonexistantFolderError


def check_correct_flag_input_val(inputs):
    flag = inputs[2]
    if (flag != "1") and (flag != "0"):
        raise UnexpectedFlagError


def check_valid_inputs(inputs):
    if len(inputs) < 2:
        raise MissingInputError
    elif len(inputs) > 3:
        raise ExcessiveInputError


def load_data(inputs):
    data_direc = inputs[1]
    for root, dirs, files in os.walk(data_direc):
        for f in files:
            file_path = os.path.join(root, f)
            raw_data = pd.read_csv(file_path)
    return raw_data


def get_flag(inputs):
    flag = inputs[2]
    if flag == "1":
        flag = 1
    elif flag == "0":
        flag = 0
    return flag


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


def get_top_n_coef_multiclass(feature_list, coef_list, n=5):
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


def top_coefs_multiclass_report(top_n_coefs, id_targ_map):
    str_report = ""
    for class_id in range(len(top_n_coefs)):
        feature_top_n = top_n_coefs[class_id]
        class_name = id_targ_map[class_id]
        for feature in feature_top_n.keys():
            coef = feature_top_n[feature]
            str_report += f"Class: {class_name} | {feature}: {coef:.4f}\n"
        str_report += "--------------------------------\n"
    return str_report


def log_section(log_file, title="No Title", content="No Content"):
    log_file.write(title + "\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    log_file.write(content + "\n")
    log_file.write("\n")


if __name__ == '__main__':
    inputs = sys.argv
    check_valid_inputs(inputs)
    check_correct_file_structure(inputs)
    full_run_flag = 0
    if len(inputs) == 3:
        check_correct_flag_input_val(inputs)
        full_run_flag = get_flag(inputs)
    else:
        info_message = "This script by default runs on a small sample size"
        info_message += " by default to save"
        info_message += " computation time and memory.\n"
        info_message += " A full run can be computed by inputting 1 as"
        info_message += " the second script argument"
        print(info_message)

    if full_run_flag:
        print("Performing a full run...")
    else:
        print("Performing an example run...")

    # Mappings
    action_to_id = {"allow": 0, "deny": 1, "drop": 2, "reset-both": 3}
    id_to_action = {0: "allow", 1: "deny", 2: "drop", 3: "reset-both"}

    # Log Start
    log_file = open("log.txt", "w")
    log_file.write("Case Study 5 Report\n\n")

    # Use input to load data
    raw_data = load_data(inputs)

    # Shuffle Data and sample
    if full_run_flag:
        raw_data = raw_data.sample(frac=1)
    else:
        raw_data = raw_data.sample(frac=0.07)

    # Preprocessing
    clean_df = raw_data.copy()
    categ_features = ["Source Port",
                      "Destination Port",
                      "NAT Source Port",
                      "NAT Destination Port"]
    target_feature = ["Action"]
    cts_features = ["Bytes",
                    "Bytes Sent",
                    "Bytes Received",
                    "Packets",
                    "Elapsed Time (sec)",
                    "pkts_sent",
                    "pkts_received"]
    scaler = StandardScaler()
    clean_df[cts_features] = scaler.fit_transform(clean_df[cts_features])
    y = clean_df[target_feature[0]]
    x = clean_df.drop(target_feature, axis=1)
    for categ_feature in categ_features:
        x[categ_feature] = x[categ_feature].astype("category")
    x = pd.get_dummies(x)

    # Split emulating a 5 fold cv
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # SGD
    SGD_params = {
        "loss": "log_loss",
        "penalty": "elasticnet",
        "early_stopping": True,
        "n_iter_no_change": 5,
        'l1_ratio': 1.0,
        'alpha': 6.30957344480193e-06
    }
    sgd_clf = SGDClassifier(**SGD_params)
    print("SGD TRAIN START")
    sgd_clf.fit(x_train, y_train)
    print("SGD TRAIN END")

    # Classif Report SGD
    y_pred_final = sgd_clf.predict(x_test)
    classif_rep = classification_report(
        y_test,
        y_pred_final
    )
    log_section(log_file,
                title="SGD Classification Report",
                content=classif_rep)

    # Confusion Matrix SGD
    conf_mat = confusion_matrix(y_test, y_pred_final)

    base_labels = y_test.unique()
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
                 title="SGD Confusion Matrix Heatmap",
                 cbar_label="Counts",
                 save_name="SGD_conf_heatmap.png")

    # Feature Importance SGD
    class_top_n = get_top_n_coef_multiclass(
        list(x.columns),
        sgd_clf.coef_,
        n=5)
    coefs_rep = top_coefs_multiclass_report(
        class_top_n,
        id_to_action
    )
    log_section(log_file,
                title="SGD Important Coeficient Report",
                content=coefs_rep)

    # SVM
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.85)
    SVC_params = {'kernel': 'poly',
                  'gamma': 'scale',
                  'degree': 3,
                  'C': 1000000.0}
    svm_clf = SVC(**SVC_params)
    print("SVM TRAIN START")
    svm_clf.fit(x_train, y_train)
    print("SVM TRAIN END")

    # Classif Report SVM
    y_pred_final = svm_clf.predict(x_test)
    classif_rep = classification_report(
        y_test,
        y_pred_final
    )
    log_section(log_file,
                title="SVM Classification Report",
                content=classif_rep)

    # Confusion Matrix SVM
    conf_mat = confusion_matrix(y_test, y_pred_final)

    base_labels = y_test.unique()
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
                 title="SVM Confusion Matrix Heatmap",
                 cbar_label="Counts",
                 save_name="svm_conf_heatmap.png")

    # SVM Feature Importance not here cuz nonlinear model

    # Fin
    log_file.write("\n")
    log_file.close()
