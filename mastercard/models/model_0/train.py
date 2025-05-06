import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mastercard.experiment_template import Config
from mastercard.models.model_0.artifacts import Artifacts


def train_pipe(config: Config, dataset: pd.DataFrame):
    model = make_pipeline(
        StandardScaler(), LogisticRegression()
    )
    model.fit(dataset[config.columns], dataset[config.target])
    
    artifacts = {
        'model': model,
    }
    return Artifacts(model=model, features=config.columns)