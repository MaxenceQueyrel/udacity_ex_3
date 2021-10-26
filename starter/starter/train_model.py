# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from joblib import dump, load

# Add code to load in the data.
path_data = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, "data/census_clean.csv"))
path_model = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, "model/logistic_regression.pkl"))
path_scores = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, "model/logistic_regression_scores.csv"))
path_encoder = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, "model/encoder.pkl"))

data = pd.read_csv(path_data)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

label = "salary"
classes = ["<=50K", ">50K"]


def process_data(data, categorical_features, label, training=True, encoder=None, with_label=True):
    if with_label:
        y_data = data[label]
        data = data.drop(label, axis=1)
        y_data = label_binarize(y_data, classes=classes).flatten()
    else:
        y_data = None
    if training:
        encoder = OneHotEncoder()
        encoder.fit(data[categorical_features])
        dump(encoder, path_encoder)
    else:
         assert encoder is not None, "For the test processing, encoder has to be passed"
    data_encoded = encoder.transform(data[categorical_features]).toarray()
    data_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(cat_features))
    data_encoded.index = data.index
    data = data.drop(categorical_features, axis=1)
    data = data.join(data_encoded)
    return data, y_data, encoder


# Train and save a model.
def train_model(model, X_train, y_train, path_save):
    model.fit(X_train, y_train)
    dump(model, path_save)


# Evaluate the model
def evaluate_model(model, X_test, y_test, path_save=None):
    score = f1_score(y_test, model.predict(X_test), zero_division=0)
    df_res = pd.DataFrame([score], columns=["f1-score"])
    if path_save:
        df_res.to_csv(path_save)
    return score


def score_age_slicing(model, data, encoder, age_slice=30):
    data_high = data[data["age"] >= age_slice]
    data_low = data[data["age"] < age_slice]

    data_high, target_high, _ = process_data(
        data_high, categorical_features=cat_features, label=label, training=False, encoder=encoder
    )

    data_low, target_low, _ = process_data(
        data_low, categorical_features=cat_features, label=label, training=False, encoder=encoder
    )

    score_high = evaluate_model(model, data_high, target_high, None)
    score_low = evaluate_model(model, data_low, target_low, None)
    print("F1-score of the model for people >= {} years old ({} samples): {:.2f}".
          format(age_slice, data_high.shape[0], score_high))
    print("F1-score of the model for people < {} years old ({} samples): {:.2f}".
          format(age_slice, data_low.shape[0], score_low))


def score_categorical_slices(model, data, encoder):
    for feature in cat_features:
        print("\n######### Feature: {} #########".format(feature))
        for value in data[feature].unique():
            data_tmp = data[data[feature] == value]
            data_tmp, target_tmp, _ = process_data(
                data_tmp, categorical_features=cat_features, label=label, training=False, encoder=encoder
            )
            score = evaluate_model(model, data_tmp, target_tmp, None)
            print("F1-score of the model for people with {} = {} ({} samples): {:.2f}".
                  format(feature, value, data_tmp.shape[0], score))


if __name__ == "__main__":
    X_train, y_train, encoder = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    print("Training data shape: %s rows, %s columns" % (X_train.shape[0], X_train.shape[1]))

    # Proces the test data with the process_data function.
    X_test, y_test, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder
    )
    print("Test data shape: %s rows, %s columns" % (X_test.shape[0], X_test.shape[1]))

    clf = LogisticRegression()
    train_model(clf, X_train, y_train, path_model)

    score = evaluate_model(clf, X_test, y_test, path_scores)
    print("F1-score of the model: {:.2f}".format(score))

    score_age_slicing(clf, test, encoder, 30)

    score_categorical_slices(clf, test, encoder)
