import pandas as pd
from sklearn import model_selection
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    df.to_csv(config.TRAINING_FILE, index=False)
