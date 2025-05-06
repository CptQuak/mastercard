import pandas as pd
from sklearn.datasets import load_iris


def create_session(random_seed=13):
    df = load_iris(as_frame=True)["frame"]

    train_dataset, test_dataset = [], []
    for value in df["target"].unique():
        group = df[df["target"] == value].copy()

        test = group.sample(n=int(len(group) * 0.3), random_state=random_seed)
        train = group.drop(test.index)

        train_dataset.append(train), test_dataset.append(test)

    train_dataset, test_dataset = (
        pd.concat(train_dataset, ignore_index=True),
        pd.concat(test_dataset, ignore_index=True),
    )
    return train_dataset, test_dataset
