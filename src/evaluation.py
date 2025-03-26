import pandas as pd
import streamlit as st


def allow_user_correction(
    df: pd.DataFrame, text_column: str, categories: list[str]
) -> pd.DataFrame:
    """Let users correct classification results.

    Args:
        df (pd.DataFrame): The DataFrame containing the classification results.
        text_column (str): The name of the text column.
        categories (list[str]): The list of categories.

    Returns:
        pd.DataFrame: The DataFrame with the corrected classifications.
    """
    st.subheader("Review and Correct Classifications")

    display_df = df[[text_column, "category"]].copy()

    corrected_df = st.data_editor(
        display_df,
        use_container_width=True,
        column_config={
            text_column: st.column_config.TextColumn("Text", disabled=True),
            "category": st.column_config.SelectboxColumn(
                "Category", options=categories
            ),
        },
        column_order=[
            text_column,
            "category",
        ],
        hide_index=True,
    )

    corrected_df["user_corrected"] = corrected_df["category"] != df["category"]

    if "user_corrected" not in corrected_df.columns:
        corrected_df["user_corrected"] = False

    return corrected_df


def calculate_metrics(df: pd.DataFrame) -> dict | None:
    """Calculate performance metrics if user corrections are available.

    Args:
        df (pd.DataFrame): The DataFrame containing the classification results.

    Returns:
        dict | None: The metrics if user corrections are available, otherwise None.
    """
    if "user_corrected" not in df.columns or df["user_corrected"].sum() == 0:
        return None

    total_samples = len(df)
    corrected_samples = df["user_corrected"].sum()
    accuracy = 1 - (corrected_samples / total_samples)

    metrics = {
        "Total samples": total_samples,
        "Corrected samples": corrected_samples,
        "Accuracy": accuracy,
    }

    category_metrics = {}
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        cat_total = len(cat_df)
        cat_corrected = cat_df["user_corrected"].sum()
        cat_accuracy = 1 - (cat_corrected / cat_total) if cat_total > 0 else 1

        category_metrics[category] = {
            "Total": cat_total,
            "Corrected": cat_corrected,
            "Accuracy": cat_accuracy,
        }

    metrics["Per category"] = category_metrics

    return metrics
