import pandas as pd
import config

if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE)

    print(df.describe())
