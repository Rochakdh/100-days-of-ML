import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

def correlations(df):
    sns.heatmap(df.corr(),cmap="BrBG" ,center=0, square=True,annot=True)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE)
    correlations(df)
    print(df.describe())
