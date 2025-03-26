import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def load_file(uploaded_file: UploadedFile) -> pd.DataFrame | None:
    """Load data from an uploaded Excel or CSV file.

    Args:
        uploaded_file (UploadedFile): The uploaded file to load.

    Returns:
        pd.DataFrame | None: The loaded DataFrame or None if an error occurs.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading file: {e!s}")
        return None


def get_text_column(df: pd.DataFrame) -> str:
    """Let the user select which column contains the text to classify.

    Args:
        df (pd.DataFrame): The DataFrame to select the text column from.

    Returns:
        str: The name of the text column.
    """
    text_column = st.selectbox(
        "Select the column containing text to classify:", options=df.columns.tolist()
    )
    return text_column


def sample_data(df: pd.DataFrame) -> pd.DataFrame:
    """Display a sample of the data for the user to review.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        text_column (str): The name of the text column.
        n (int): The number of rows to sample.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    st.subheader("Sample Data")
    st.write(df.head())

    return df
