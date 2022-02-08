import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import config
import argparse
import model_dispatcher
import os

cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}
def normalize(df):
    result = df.copy()
    for features_name in df.columns:
        max_value = df[features_name].max()
        min_value = df[features_name].min()
        result[features_name] = (df[features_name]-min_value) / (max_value - min_value)
    return pd.DataFrame(result)

def run(fold,model):
    df = pd.read_csv(config.TRAINING_FILE)

    df.loc[:, "quality"] = df.quality.map(quality_mapping)
    df = df.sample(frac=1).reset_index(drop=True)


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)


    """
        clf.fit(df_train[cols], df_train.quality)
    """
    x_train = df_train.drop(["quality","kfold"], axis=1)
    x_train = normalize(x_train)
    x_train = x_train.values

    y_train = df_train.quality.values

    x_valid = df_valid.drop(["quality","kfold"], axis=1)
    x_valid = normalize(x_valid)
    x_valid = x_valid.values

    y_valid = df_valid.quality.values

    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold},Accuracy ={accuracy}")

    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"modelsdt_{fold}.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    run(
        fold=args.fold,
        model = args.model
    )
