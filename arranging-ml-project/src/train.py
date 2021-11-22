import pandas as pd
from sklearn.externals import joblib
from sklearn import tree
from sklearn import metrics

import config


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold},Accuracy ={accuracy}")

    joblib.dump(clf, f"../modelsdt_{fold}.bin")


if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)
