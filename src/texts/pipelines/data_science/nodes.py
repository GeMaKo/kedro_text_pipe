import logging
from time import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

logger = logging.getLogger(__name__)


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)


def tfidf_transform(X: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Create a TSNE-Visualization of text data"""
    tfidf = TfidfVectorizer(**parameters)
    t0 = time()
    X_tfidf = tfidf.fit_transform(X["message"])
    logger.info(f"vectorization done in {time() - t0:.3f} s")
    logger.info(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")
    return X_tfidf, tfidf


def lsa(X_tfidf: pd.DataFrame) -> pd.DataFrame:

    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    t0 = time()
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    logger.info(f"LSA done in {time() - t0:.3f} s")
    logger.info(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
    return X_lsa, lsa


def kmeans_clustering(X: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    kmeans = KMeans(**parameters).fit(X)
    cluster_labels = pd.DataFrame(kmeans.labels_)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    cluster_info = pd.DataFrame(
        {"Cluster_IDs": cluster_ids, "Cluster_sizes": cluster_sizes}
    )
    return cluster_labels, cluster_info, kmeans


def tsne_transform(X: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Create a TSNE-Embedding of the data"""
    tsne = TSNE(**parameters)
    X_tsne = tsne.fit_transform(X)
    return pd.DataFrame(X_tsne)


def top_terms_per_cluster(vectorizer, lsa, kmeans, parameters: Dict):
    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(parameters["n_clusters"]):
        top_terms = str([terms[ind] for ind in order_centroids[i, :10]])
        logger.info(f"Cluster {i}: {top_terms}")
