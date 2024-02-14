import re
from typing import Dict, Tuple

import pandas as pd


def read_mails(emails) -> pd.DataFrame:
    """Combine the texts into a single DataFrame"""
    email_data = emails["file"].str.split("/", expand=True)
    email_data["message"] = emails["message"]
    return email_data


def process_email_folders(email_data) -> pd.DataFrame:
    """Rename and Drop columns from email DataFrame"""
    email_data.rename(columns={0: "name", 1: "top_folder"}, inplace=True)
    print(email_data.columns)
    email_data.drop(columns=[2, 3, 4, 5, 6], inplace=True)
    return email_data


def sample_data(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Create a sample dataset"""
    sampled_data = data.sample(**parameters)
    return sampled_data
