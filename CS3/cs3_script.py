import os
import sys
import email
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics as mt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# Build a spam classifier using naive Bayes and clustering.
# You will have to create your own dataset from the input messages.
# Be sure to document how you created your dataset.

INVALID_INPUT = 1


class UnexpectedEmailFileStructureError(Exception):
    def __init__(self):
        msg = "This script expects a file path to a directory that contains "
        msg += "the spam assassin dataset. The spam assassin dataset is a "
        msg += "group of five folders with the names, easy_ham, easy_ham_2, "
        msg += "hard_ham, spam, spam_2. Did you specify a folder path that "
        msg += "contains these folders?"
        super().__init__(msg)


class MissingInputError(Exception):
    def __init__(self):
        msg = "This script expects exactly one input, a file directory that "
        msg += "contains the spam assassin dataset. You input nothing. "
        msg += "Did you forget to add an argument with the script?"
        super().__init__(msg)


class ExcessiveInputError(Exception):
    def __init__(self):
        msg = "This script expects atleast one input, a file directory that "
        msg += "contains the spam assassin dataset. You called this script "
        msg += "with more than one argument. Only use one and try again. "
        super().__init__(msg)


def recursive_payload_retrieval(email_payload):
    email_body = ""
    for sub_email in email_payload:
        sub_email_payload = sub_email.get_payload()
        if type(sub_email_payload) is list:
            email_body += recursive_payload_retrieval(sub_email_payload)
        elif type(sub_email_payload) is str:
            email_body += sub_email_payload
    return email_body


def check_correct_file_structure(inputs):
    email_direc = inputs[1]
    expected_direcs = ["easy_ham", "easy_ham_2", "hard_ham", "spam", "spam_2"]
    first = True
    for root, dirs, files in os.walk(email_direc):
        if first:
            first = False
            for expected_direc in expected_direcs:
                if expected_direc not in dirs:
                    print(expected_direc)
                    print(dirs)
                    raise UnexpectedEmailFileStructureError


def check_valid_inputs(inputs):
    if len(inputs) < 2:
        raise MissingInputError
    elif len(inputs) > 2:
        raise ExcessiveInputError


def load_emails(inputs):
    email_direc = inputs[1]
    data = {"email_body": [], "email_type": [], "email_class": []}
    for root, dirs, files in os.walk(email_direc):
        for f in files:
            file_path = os.path.join(root, f)
            with open(file_path, "r", encoding="latin-1") as email_file:
                if "cmds" in file_path:
                    continue
                body = ""
                msg = email.message_from_file(email_file)
                t = msg.get_content_type()
                payload = msg.get_payload()
                if type(payload) is list:
                    body = recursive_payload_retrieval(payload)
                elif type(payload) == str:
                    body = payload
                else:
                    print("Unexpected Condition")
                    raise Exception()

                data["email_type"].append(t)

                data["email_body"].append(body)

                if "spam" in root:
                    data["email_class"].append("spam")
                else:
                    data["email_class"].append("not_spam")
    return data


def categ_categ_compare(df, categ1, categ2, file_name="no_name.png"):
    horizontal = 1
    vertical = 0
    df_count = df.groupby([categ1, categ2])[categ1].count().unstack().fillna(0)
    df_relative_count = df_count.div(
        df_count.sum(axis=horizontal),
        axis=vertical)*100
    title = "Percentage Composition of " + categ1 + " with " + categ2
    df_relative_count.plot.barh(stacked=True, title=title)
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()


def multinom_nb_cv(x, y, print_progress=False, n_jobs=1):
    best_alpha = 0
    best_f1 = 0
    first = True
    iter_count = 20
    report_freq = int(iter_count/20)
    i = 0
    t1 = time.time()
    for param_alpha in np.logspace(-3, 3, iter_count):

        model = MultinomialNB(alpha=param_alpha)

        f1_cv = cross_val_score(model, x, y, cv=5,
                                scoring='f1')

        f1 = sum(f1_cv)/len(f1_cv)

        if first:
            first = False
            best_alpha = param_alpha
            best_f1 = f1
        else:
            if (f1 > best_f1):
                best_alpha = param_alpha
                best_f1 = f1

        if print_progress and ((i+1) % report_freq == 0):
            print(f"Iteration {i+1}. "
                  f"Percent done = {(i+1)/iter_count*100:.4f}%")

        i += 1
    t2 = time.time()
    elapsed_time = t2-t1

    report_str = f"Best f1={best_f1:.4f}, "
    report_str += f"alpha={best_alpha}, time={elapsed_time:.4f}s\n"

    return best_alpha, report_str


def log_section(title="No Title", content="No Content"):
    log_file.write(title + "\n")
    log_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    log_file.write(content + "\n")
    log_file.write("\n")


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
    inputs = sys.argv
    check_valid_inputs(inputs)
    check_correct_file_structure(inputs)
    # Log Start
    log_file = open("log.txt", "w")
    log_file.write("Case Study 3 Report\n\n")

    # Use input to find and load emails
    email_data_raw = load_emails(inputs)
    email_df = pd.DataFrame(email_data_raw)

    # Id to class maps
    email_class_to_id = {"spam": 1, "not_spam": 0}
    id_to_email_class = {1: "spam", 0: "not_spam"}

    # TFIDF Vecorizer
    email_bodies = email_df["email_body"]
    vectorizer = TfidfVectorizer(min_df=0.002)
    features = vectorizer.fit_transform(email_bodies)

    # Cluster with Kmeans
    cluster_model = KMeans(n_clusters=2)
    cluster_model.fit(features)
    email_df["cluster_id"] = cluster_model.labels_

    # Plot to see which clusters refers to what email class
    categ_categ_compare(email_df,
                        "email_class",
                        "cluster_id",
                        "categorical_cluster_comparison.png")

    # add cluster feature to vectorizer features
    cluster_ids_raw = cluster_model.labels_
    cluster_ids_reshaped = cluster_ids_raw.reshape(-1, 1)
    features_and_cluster_ids = hstack((features, cluster_ids_reshaped)).A

    # Optimizing Alpha
    email_id = email_df["email_class"].map(email_class_to_id)
    best_alpha, report = multinom_nb_cv(features_and_cluster_ids,
                                        email_id,
                                        n_jobs=5,
                                        print_progress=True)
    log_section(title="Optimization Report",
                content=report)

    # Best Model
    final_clf = MultinomialNB(alpha=best_alpha)
    final_clf.fit(features_and_cluster_ids, email_df["email_class"])

    # Classification Report
    y_pred_final = final_clf.predict(features_and_cluster_ids)
    y_true = email_df["email_class"]
    classif_rep = classification_report(
        y_true,
        y_pred_final
    )
    log_section(title="Classification Report",
                content=classif_rep)

    # Confusion Matrix Heatmap
    conf_mat = confusion_matrix(y_true, y_pred_final)

    base_labels = list(id_to_email_class.values())
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

    # ROC Curve
    y_pred_prob = final_clf.predict_proba(features_and_cluster_ids)
    y_label = id_to_email_class[1]
    preds = y_pred_prob[:, 1]
    fpr, tpr, thresholds = mt.roc_curve(email_id, preds, pos_label=i)
    plt.plot(fpr, tpr, label=y_label)
    title_str = 'ROC Curve'
    plt.title(title_str)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    auc_report = f"{y_label} AUC: {mt.auc(fpr, tpr):.4f}\n"
    log_section(title="AUC Report",
                content=auc_report)
    plt.legend()
    plt.savefig("roc_plot.png", bbox_inches='tight')
    plt.clf()
    log_file.write("\n")
    log_file.close()
