import asyncio
import json
import time

import pandas as pd
import streamlit as st
from src.classification import TextClassifier
from src.core.config import settings
from src.evaluation import allow_user_correction, calculate_metrics
from src.explanation import show_detailed_results
from src.utils.data import get_text_column, load_file, sample_data


def main():
    """Run the main application"""

    st.set_page_config(
        page_title="Text Classification System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "classifier" not in st.session_state:
        st.session_state.classifier = TextClassifier(model=settings.MODEL)
    if "categories" not in st.session_state:
        st.session_state.categories = {}
    if "predefined_options" not in st.session_state:
        with open("data/categories.json") as f:
            st.session_state.predefined_options = json.load(f)
    if "data_df" not in st.session_state:
        st.session_state.data_df = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "text_column" not in st.session_state:
        st.session_state.text_column = None

    st.title("🔍 Text Classification System")
    st.write("""
    Upload your data, define categories, and use AI to classify your text data.
    No technical expertise required!
    """)

    st.header("Step 1: Upload Your Data")

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file:", type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:
        df = load_file(uploaded_file)

        if df is not None:
            st.session_state.data_df = df

        st.session_state.text_column = get_text_column(df)

        sample_data(df)

    if st.session_state.data_df is not None:
        st.header("Step 2: Define Your Categories")

        predefined = st.checkbox("Use predefined category examples")

        if predefined:
            category_set = st.selectbox(
                "Select category set:", list(st.session_state.predefined_options.keys())
            )

            st.session_state.categories = st.session_state.predefined_options[
                category_set
            ]

            st.subheader(f"{category_set} Categories:")
            for cat, desc in st.session_state.categories.items():
                st.markdown(f"- **{cat}**: {desc}")

        else:
            st.write("Define your own categories by adding them below:")

            if not st.session_state.categories:
                st.session_state.categories = {"": ""}

            categories_to_remove = []
            new_categories = {}

            for cat, desc in st.session_state.categories.items():
                col1, col2, col3 = st.columns([3, 6, 1])
                new_cat = col1.text_input("Category:", value=cat, key=f"cat_{cat}")
                new_desc = col2.text_input(
                    "Description:", value=desc, key=f"desc_{cat}"
                )
                if col3.button("✖", key=f"del_{cat}"):
                    categories_to_remove.append(cat)
                else:
                    new_categories[new_cat] = new_desc

            for cat in categories_to_remove:
                if cat in st.session_state.categories:
                    del st.session_state.categories[cat]
                    st.rerun()

            if not categories_to_remove:
                st.session_state.categories = {
                    k: v for k, v in new_categories.items() if k
                }

            if st.button("+ Add Category"):
                st.session_state.categories[""] = ""

        if st.session_state.classifier is not None and st.session_state.categories:
            st.session_state.classifier.set_categories(st.session_state.categories)

    if st.session_state.data_df is not None and st.session_state.categories:
        st.header("Step 3: Classify Your Texts")

        if st.button("Start Classification") and st.session_state.text_column:
            texts = st.session_state.data_df[st.session_state.text_column].tolist()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing... {int(progress * 100)}% complete")

            start_time = time.time()
            status_text.text("Starting classification...")

            loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                st.session_state.classifier.batch_classify(texts, update_progress)
            )

            loop.close()

            results_df = st.session_state.data_df.copy()
            results_df["original_text"] = results_df[st.session_state.text_column]
            results_df["category"] = [
                r.category if r.category else "Unknown" for r in results
            ]
            results_df["confidence"] = [
                r.confidence if r.confidence else 0 for r in results
            ]
            results_df["explanation"] = [
                r.explanation if r.explanation else "" for r in results
            ]
            results_df["keywords"] = [r.keywords if r.keywords else [] for r in results]
            results_df["ambiguities"] = [
                r.ambiguities if r.ambiguities else [] for r in results
            ]

            st.session_state.results_df = results_df

            elapsed_time = time.time() - start_time
            status_text.text(
                f"Classification complete! Processed {len(texts)} texts in {elapsed_time:.2f} seconds."
            )
            progress_bar.progress(1.0)

        if st.session_state.results_df is not None:
            st.header("Classification Results")

            show_detailed_results(
                st.session_state.results_df, st.session_state.text_column
            )

            st.subheader("Data visualization")
            col1, col2 = st.columns(2)

            with col1:
                category_counts = st.session_state.results_df["category"].value_counts()
                st.write("Category distribution:")
                st.bar_chart(category_counts)

            with col2:
                avg_confidence = st.session_state.results_df.groupby("category")[
                    "confidence"
                ].mean()
                st.write("Average confidence by category:")
                st.bar_chart(avg_confidence)

            st.header("Step 4: Review and Improve")

            if st.checkbox("Enable correction mode"):
                corrected_df = allow_user_correction(
                    st.session_state.results_df,
                    st.session_state.text_column,
                    ["Undetermined", *st.session_state.categories.keys()],
                )

                if st.button("Save Corrections"):
                    st.session_state.results_df = corrected_df
                    st.success("Corrections saved!")

                    metrics = calculate_metrics(corrected_df)
                    if metrics:
                        st.subheader("Model Performance Metrics")
                        st.write(f"**Overall accuracy**: {metrics['Accuracy']:.2%}")
                        st.write(f"**Total samples**: {metrics['Total samples']}")
                        st.write(
                            f"**Corrected samples**: {metrics['Corrected samples']}"
                        )

                        st.subheader("Per-Category Performance")
                        metrics_df = pd.DataFrame.from_dict(
                            {
                                k: {
                                    "Total": v["Total"],
                                    "Accuracy": f"{v['Accuracy']:.2%}",
                                }
                                for k, v in metrics["Per category"].items()
                            },
                            orient="index",
                        )

                        st.dataframe(metrics_df)

            st.download_button(
                "Download Results as CSV",
                st.session_state.results_df.to_csv(index=False).encode("utf-8"),
                "classification_results.csv",
                "text/csv",
                key="download-csv",
            )

    st.markdown("---")
    st.markdown("### About This Tool")
    st.write("""
    This text classification system uses advanced AI to help non-technical users classify their text data.
    Simply upload your data, define categories, and let the system do the work.
    The system provides transparent explanations for each classification.
    """)


if __name__ == "__main__":
    main()
