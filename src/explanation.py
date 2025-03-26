import pandas as pd
import streamlit as st


def show_detailed_results(df: pd.DataFrame, text_column: str) -> None:
    """Show detailed classification results with options to filter.

    Args:
        df (pd.DataFrame): The dataframe to show.
        text_column (str): The column to show.
    """
    st.subheader("Detailed Classification Results")

    unique_categories = sorted(df["category"].unique().tolist())
    all_categories = ["All", *unique_categories]

    selected_category = st.selectbox("Filter by category :", all_categories)

    confidence_threshold = st.slider("Minimum confidence score :", 0.0, 1.0, 0.5, 0.1)

    filtered_df = df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    filtered_df = filtered_df[filtered_df["confidence"] >= confidence_threshold]

    if not filtered_df.empty:
        st.write(f"Total results: {len(filtered_df)}")

        display_df = filtered_df[
            [text_column, "category", "confidence", "explanation"]
        ].copy()

        if "keywords" in filtered_df.columns and any(filtered_df["keywords"].notna()):
            display_df["keywords"] = (
                filtered_df["keywords"] if len(filtered_df["keywords"]) > 0 else None
            )

        if "ambiguities" in filtered_df.columns and any(
            filtered_df["ambiguities"].notna()
        ):
            display_df["ambiguities"] = (
                filtered_df["ambiguities"]
                if len(filtered_df["ambiguities"]) > 0
                else None
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                text_column: st.column_config.TextColumn("Text"),
                "category": st.column_config.TextColumn("Category"),
                "confidence": st.column_config.NumberColumn("Confidence"),
                "explanation": st.column_config.TextColumn(
                    "Explanation", width="large"
                ),
                "keywords": st.column_config.ListColumn("Keywords"),
                "ambiguities": st.column_config.JsonColumn(
                    "Ambiguities", width="medium"
                ),
            },
            hide_index=True,
        )

    else:
        st.write("No results match the selected filters.")
