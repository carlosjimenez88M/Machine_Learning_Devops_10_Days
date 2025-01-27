'''
    Preprocessing Step
    Release Date:2025-01-26
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import os
import argparse
import logging
import wandb
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

#================================#
# ---- logger configuration ---- #
#================================#

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")

logger = logging.getLogger()

#=========================#
# ---- main function ---- #
#=========================#

def go(args):
    logger.info("Initializing Preprocessing Steps...")
    run = wandb.init(job_type="preprocess_data")
    logger.info('Downloading dataset...')
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    logger.info('Transforming data...')
    target_column = "median_house_value"
    if target_column not in df.columns:
        logger.error(f"The target column '{target_column}' is missing from the dataset.")
        raise KeyError(f"The target column '{target_column}' is missing from the dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])  # Resto de las columnas

    # Definir columnas numéricas y categóricas
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                   "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]

    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
            self.n_clusters = n_clusters
            self.gamma = gamma
            self.random_state = random_state

        def fit(self, X, y=None, sample_weight=None):
            self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
            self.kmeans_.fit(X, sample_weight=sample_weight)
            return self

        def transform(self, X):
            cluster_labels = self.kmeans_.predict(X)
            return np.expand_dims(cluster_labels, axis=1)  # Devuelve una sola columna con las etiquetas de cluster

        def get_feature_names_out(self, names=None):
            return ["cluster_label"]

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ("geo", ClusterSimilarity(n_clusters=10, gamma=0.1, random_state=42), ["latitude", "longitude"]),
    ])

    logger.info("Applying transformations...")
    processed_data = preprocessing.fit_transform(X)

    num_features = num_attribs
    cat_features = preprocessing.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_attribs)
    geo_features = preprocessing.named_transformers_["geo"].get_feature_names_out()
    all_features = np.concatenate([num_features, cat_features, geo_features])

    df_processed = pd.DataFrame(processed_data, columns=all_features, index=X.index)

    # Añadir la columna objetivo al conjunto de datos procesado
    df_processed[target_column] = y

    logger.info("Saving preprocessed dataset...")
    df_processed.to_csv(args.output_artifact, index=False)
    logger.info(df_processed.columns)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.output_artifact)

    logger.info("Logging artifact...")
    run.log_artifact(artifact)
    run.finish()

    logger.info("Preprocessing completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess the dataset")

    parser.add_argument('--input_artifact',
                        type=str,
                        required=True,
                        help="Input artifact name")

    parser.add_argument('--artifact_name',
                        type=str,
                        required=True,
                        help="Output artifact name")

    parser.add_argument('--artifact_description',
                        type=str,
                        required=True,
                        help="Description of the output artifact")

    parser.add_argument('--artifact_type',
                        type=str,
                        required=True,
                        help="Type of the output artifact")

    parser.add_argument('--output_artifact',
                        type=str,
                        required=True,
                        help="Path to save the preprocessed dataset")

    args = parser.parse_args()
    go(args)
