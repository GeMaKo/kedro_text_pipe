from kedro.pipeline import Pipeline, node, pipeline

from .nodes import process_email_folders, read_mails, sample_data


def create_pipeline(**kwargs) -> Pipeline:
    data_processing_pipe = pipeline(
        [
            node(
                func=read_mails,
                inputs="emails",
                outputs="email_data",
                name="read_mails_node",
            ),
            node(
                func=sample_data,
                inputs=["email_data", "params:sampling"],
                outputs="sampled_data",
                name="sample_data_node",
            ),
            node(
                func=process_email_folders,
                inputs="sampled_data",
                outputs="proc_email_data",
                name="process_email_folders_node",
            ),
        ]
    )
    return data_processing_pipe
