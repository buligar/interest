import os
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    # create folds
    df = pd.read_csv("D:/Downloads/ISIC_2020_Training_GroundTruth_v2.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv("D:/Downloads/ISIC_2020_Training_GroundTruth_v2.csv", index=False)


