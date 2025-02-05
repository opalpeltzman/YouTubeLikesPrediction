import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn import svm as skl_svm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance


from matplotlib import pyplot
from xgboost import plot_importance

IRIS_URL = "https://raw.githubusercontent.com/opalpeltzman/YouTubeLikesPrediction/main/final_project/datasets/IRIS.csv"
TITANIC_URL = "https://raw.githubusercontent.com/opalpeltzman/YouTubeLikesPrediction/main/final_project/datasets/titanic.csv"
CANCER_URL = "https://raw.githubusercontent.com/opalpeltzman/YouTubeLikesPrediction/main/final_project/datasets/cancer.csv"
SLEEPING_URL = "https://raw.githubusercontent.com/opalpeltzman/YouTubeLikesPrediction/main/final_project/datasets/sleeping.csv"
IRIS_DATASET = "iris_dataset"
TITANIC_DATASET = "titanic_dataset"
CANCER_DATASET = "cancer_dataset"
SLEEPING_DATASET = "sleeping_dataset"

shap.initjs()
SEED = 1337
np.random.seed(SEED)


def reload_datasets():
    return {
        IRIS_DATASET: pd.read_csv(IRIS_URL),
        TITANIC_DATASET: pd.read_csv(TITANIC_URL),
        SLEEPING_DATASET: pd.read_csv(SLEEPING_URL),
        CANCER_DATASET: pd.read_csv(CANCER_URL),
    }


def preprocess_dataset(name, dataset):
    """Preprocess different datasets by selecting relevant features and target variables.

    Args:
        name (str): Name of the dataset to preprocess
        dataset (pd.DataFrame): Input dataset

    Returns:
        tuple: (X, Y) where X contains features and Y contains target variable
    """
    preprocessing_configs = {
        "iris_dataset": {
            "target": "species",
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        },
        "titanic_dataset": {
            "target": "Survived",
            "features": [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Cabin",
                "Embarked",
            ],
        },
        "sleeping_dataset": {
            "target": "sl",
            "features": ["sr1", "rr", "t", "lm", "bo", "rem", "sr2", "hr"],
            "preprocessing": lambda df: df.assign(sl=df["sl"].astype(int)),
        },
        "cancer_dataset": {
            "target": "diagnosis",
            "features": [
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave points_mean",
                "symmetry_mean",
                "fractal_dimension_mean",
                "radius_se",
                "texture_se",
                "perimeter_se",
                "area_se",
                "smoothness_se",
                "compactness_se",
                "concavity_se",
                "concave points_se",
                "symmetry_se",
                "fractal_dimension_se",
                "radius_worst",
                "texture_worst",
                "perimeter_worst",
                "area_worst",
                "smoothness_worst",
                "compactness_worst",
                "concavity_worst",
                "concave points_worst",
                "symmetry_worst",
                "fractal_dimension_worst",
            ],
        },
    }

    if name not in preprocessing_configs:
        raise ValueError(f"Unknown dataset name: {name}")

    config = preprocessing_configs[name]

    # Apply any preprocessing steps
    if "preprocessing" in config:
        dataset = config["preprocessing"](dataset)

    X = dataset[config["features"]]
    Y = dataset[config["target"]]

    return X, Y


def KNN(X_train, Y_train, X_test, dataset_name, class_names):
    """Train KNN model, generate SHAP values and predictions"""
    print(f"Executing KNN on {dataset_name}")
    plt.subplot(121)

    # Train model
    knn = neighbors.KNeighborsClassifier(n_neighbors=15, weights="distance")
    knn.fit(X_train.values, Y_train)

    # Generate SHAP values
    knn_explainer = shap.KernelExplainer(knn.predict, X_test)
    knn_shap_values = knn_explainer.shap_values(X_test)

    # Plot and save SHAP summary
    shap.summary_plot(knn_shap_values, X_test, show=False)
    plt.savefig(f"./graphs/{dataset_name}_KNN.png")

    return knn.predict(X_test)


def SVM(X_train, Y_train, X_test, dataset_name, class_names):
    """Train SVM model, generate SHAP values and predictions"""
    print(f"Executing SVM on {dataset_name}")
    plt.subplot(121)

    # Train model
    svm = skl_svm.SVC(gamma="scale", decision_function_shape="ovo")
    svm.fit(X_train.values, Y_train)

    # Generate SHAP values
    svm_explainer = shap.KernelExplainer(svm.predict, X_test)
    svm_shap_values = svm_explainer.shap_values(X_test)

    # Plot and save SHAP summary
    shap.summary_plot(svm_shap_values, X_test, show=False)
    plt.savefig(f"./graphs/{dataset_name}_SVM.png")

    return svm.predict(X_test)


def Decision_Tree(X_train, Y_train, X_test, dataset_name, class_names):
    """Train Decision Tree model, generate SHAP values and predictions"""
    print(f"Executing Decision_Tree on {dataset_name}")

    # Train model
    model = DecisionTreeClassifier(random_state=1, max_depth=5)
    model.fit(X_train, Y_train)

    # Generate SHAP values
    decision_tree_explainer = shap.TreeExplainer(model)
    decision_tree_shap_values = decision_tree_explainer.shap_values(X_test)

    # Plot and save SHAP summary
    shap.summary_plot(
        decision_tree_shap_values, X_test, show=False, class_names=class_names
    )
    plt.savefig(f"./graphs/{dataset_name}_decision_tree.png")

    return model.predict(X_test)


def Logistic_Regression(X_train, Y_train, X_test, dataset_name, class_names):
    """Train Logistic Regression model, generate SHAP values and predictions"""
    print(f"Executing Logistic_Regression on {dataset_name}")

    # Train model
    logistic_reg_model = LogisticRegression(solver="lbfgs")
    logistic_reg_model.fit(X_train, Y_train)

    # Generate SHAP values
    logistic_reg_explainer = shap.KernelExplainer(logistic_reg_model.predict, X_test)
    logistic_reg_shap_values = logistic_reg_explainer.shap_values(X_test)

    # Plot and save SHAP summary
    shap.summary_plot(logistic_reg_shap_values, X_test, show=False)
    plt.savefig(f"./graphs/{dataset_name}_logistic_regression.png")

    return logistic_reg_model.predict(X_test)


def XGBoostClassifier(X_train, Y_train, dataset_name):
    """Train XGBoost model and plot feature importance"""
    print(f"Executing XGBoostClassifier on {dataset_name}")

    # Train model
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, Y_train)

    # Plot and save feature importance
    plot_importance(xgb_model)
    pyplot.show()
    plt.savefig(f"./graphs/{dataset_name}_XGBOOST.png")

    return xgb_model.predict(X_train)


models = {
    "KNN": lambda X_train, Y_train, X_test, dataset_name, class_names: KNN(
        X_train, Y_train, X_test, dataset_name, class_names
    ),
    "SVM": lambda X_train, Y_train, X_test, dataset_name, class_names: SVM(
        X_train, Y_train, X_test, dataset_name, class_names
    ),
    "Decision_Tree": lambda X_train, Y_train, X_test, dataset_name, class_names: Decision_Tree(
        X_train, Y_train, X_test, dataset_name, class_names
    ),
    "Logistic_Regression": lambda X_train, Y_train, X_test, dataset_name, class_names: Logistic_Regression(
        X_train.values, Y_train, X_test, dataset_name, class_names
    ),
}


shap.initjs()
np.random.seed(0)


def models_train(X_train, y_train, X_test, y_test, dataset_name, target_strings):
    """Train and evaluate multiple models on the given dataset.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of the dataset being used
        target_strings: List of target class names

    Returns:
        None. Prints classification reports and displays plots for each model.
    """
    for model_name, model_fn in models.items():
        # Train model and get predictions
        y_predict = model_fn(X_train, y_train, X_test, dataset_name, target_strings)

        # Print classification metrics
        print(f"\nResults for {model_name}:")
        print(classification_report(y_test, y_predict, target_names=target_strings))

        # Show generated plots
        plt.show()


def plot_pie_train_test(y_train, y_test):
    plt.figure(figsize=(10, 15))

    plt.subplot(121)
    plt.pie(y_train.value_counts(), labels=y_train.unique(), autopct="%1.2f%%")
    plt.title("Training Dataset")

    plt.subplot(122)
    plt.pie(y_test.value_counts(), labels=y_test.unique(), autopct="%1.2f%%")
    plt.title("Test Dataset")

    plt.tight_layout()
    plt.show()


def encode_labels(y_train, y_test, Y, name):
    le = preprocessing.LabelEncoder()
    trained_le = le.fit(y_train)
    y_train = trained_le.transform(y_train)
    y_test = trained_le.transform(y_test)
    if name == IRIS_DATASET:
        target_strings = le.inverse_transform(np.arange(len(Y.unique())))
    if name == TITANIC_DATASET:
        target_strings = np.array(["Not Survived", "Survived"])
    if name == SLEEPING_DATASET:
        target_strings = np.array(["3", "1", "0", "2", "4"])
    if name == CANCER_DATASET:
        target_strings = Y.unique()
    return y_train, y_test, target_strings


def feature_selection(X_train, X_test, name):
    if name == "iris_dataset":
        X_train = X_train.drop(columns=["sepal_length"])
        X_test = X_test.drop(columns=["sepal_length"])
    elif name == "titanic_dataset":
        X_train = X_train.drop(columns=["Parch"])
        X_test = X_test.drop(columns=["Parch"])
    elif name == "sleeping_dataset":
        X_train = X_train.drop(columns=["rr"])
        X_test = X_test.drop(columns=["rr"])
    elif name == "cancer_dataset":
        X_train = X_train.drop(
            columns=[
                "compactness_worst",
                "concave points_se",
                "concavity_se",
                "fractal_dimension_se",
                "radius_mean",
                "smoothness_worst",
                "symmetry_worst",
            ]
        )
        X_test = X_test.drop(
            columns=[
                "compactness_worst",
                "concave points_se",
                "concavity_se",
                "fractal_dimension_se",
                "radius_mean",
                "smoothness_worst",
                "symmetry_worst",
            ]
        )
    return X_train, X_test


def check_missing_values(X, name):
    print(X.isnull().sum())
    if name == TITANIC_DATASET:
        X.loc[:, "Age"] = X["Age"].fillna(X["Age"].mean())
        X.loc[:, "Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
        X = X.drop(["Cabin"], axis=1)
    return X


def main():
    dfs = reload_datasets()
    for dataset_name, dataset in dfs.items():
        dataset = shuffle(dataset)
        X, Y = preprocess_dataset(dataset_name, dataset)

        X = check_missing_values(X, dataset_name)
        # Encoding categorical features
        if dataset_name == TITANIC_DATASET:
            X["Sex"] = LabelEncoder().fit_transform(X["Sex"])
            X["Embarked"] = LabelEncoder().fit_transform(X["Embarked"])

        # Split dataset to train and test with the same ratio as before
        plt.pie(Y.value_counts(), labels=Y.unique(), autopct="%1.2f%%")
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=12, test_size=0.2)
        for train, test in splitter.split(X, Y):  # this will splits the index
            X_train = X.iloc[train]
            y_train = Y.iloc[train]
            X_test = X.iloc[test]
            y_test = Y.iloc[test]

        plot_pie_train_test(y_train, y_test)

        # Encode labels
        y_train, y_test, target_strings = encode_labels(
            y_train, y_test, Y, dataset_name
        )

        # Train the models
        models_train(X_train, y_train, X_test, y_test, dataset_name, target_strings)
        XGBoostClassifier(X_train, y_train, dataset_name)

        feature_selection(X_train, X_test, dataset_name)

        # Train the models again after feature selection with SHAP
        models_train(X_train, y_train, X_test, y_test, dataset_name, target_strings)


if __name__ == "__main__":
    main()
