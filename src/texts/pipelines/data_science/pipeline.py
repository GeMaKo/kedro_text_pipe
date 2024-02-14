from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    kmeans_clustering,
    lsa,
    tfidf_transform,
    top_terms_per_cluster,
    tsne_transform,
)


def create_pipeline(**kwargs) -> Pipeline:
    embedding_pipeline = pipeline(
        [
            node(
                func=tfidf_transform,
                inputs=["proc_email_data", "params:tfidf"],
                outputs=["X_tfidf", "tfidf"],
                name="tfidf_node",
            ),
            node(
                func=lsa,
                inputs=["X_tfidf"],
                outputs=["X_lsa", "lsa"],
                name="lsa_node",
            ),
            node(
                func=kmeans_clustering,
                inputs=["X_lsa", "params:kmeans"],
                outputs=["cluster_labels", "cluster_info", "kmeans"],
                name="kmeans_node",
            ),
            node(
                func=tsne_transform,
                inputs=["X_lsa", "params:tsne"],
                outputs="X_tsne",
                name="tsne_node",
            ),
            node(
                func=top_terms_per_cluster,
                inputs=["tfidf", "lsa", "kmeans", "params:kmeans"],
                outputs=None,
                name="cluster_report_node",
            ),
        ]
    )
    return embedding_pipeline
