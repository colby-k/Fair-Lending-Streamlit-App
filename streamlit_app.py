import textwrap  # ✅ Used for label wrapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

# ✅ Page setup with favicon
st.set_page_config(
    page_title="Fair Lending Analysis",
    page_icon="portfolio.ico",  # Make sure this is in your repo root
    layout="wide"
)

st.title("📊 Fair Lending Analysis Tool")

# Load data
@st.cache_data
def load_data():
    pricing = pd.read_csv("Pricing_data.csv")
    uw = pd.read_csv("UW_data.csv")
    return pricing, uw

pricing_df, uw_df = load_data()

# Tabs
pricing_tab, uw_tab = st.tabs(["Pricing Analysis", "Underwriting Analysis"])

# Shared options
demographic_cols = ["Race", "Sex", "Ethnicity", "Age"]

# --- Pricing Analysis ---
with pricing_tab:
    st.header("Pricing Disparity Analysis")
    demo_col = st.selectbox("Select demographic group", demographic_cols, key="pricing_demo")
    loan_types = pricing_df["LoanType"].dropna().unique().tolist()
    selected_loan = st.selectbox("Filter by Loan Type", ["All"] + loan_types, key="pricing_loan")

    df = pricing_df.copy()
    if selected_loan != "All":
        df = df[df["LoanType"] == selected_loan]

    df = df[[demo_col, "AIP"]].dropna()
    st.markdown(f"### AIP by {demo_col}")

    # Summary stats
    group_stats = df.groupby(demo_col)["AIP"].agg(["count", "mean", "std"])
    st.dataframe(group_stats)

    # Visualization: mean AIP by group with error bars
    means = group_stats["mean"]
    stds = group_stats["std"]
    overall_mean = df["AIP"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(means.index, means.values, yerr=stds.values, capsize=5, color="#6baed6")

    ax.set_title(f"Average AIP by {demo_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel(demo_col, fontsize=12)
    ax.set_ylabel("Average AIP", fontsize=12)

    # Wrap long x-axis labels
    labels = [str(label) for label in means.index]
    wrapped_labels = ["\n".join(textwrap.wrap(label, width=10)) for label in labels]
    ax.set_xticklabels(wrapped_labels, rotation=0)

    # Overall mean reference line
    ax.axhline(
        overall_mean,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall mean AIP ({overall_mean:.2f})"
    )

    # Value labels on bars
    offset = stds.max()
    if np.isnan(offset) or offset == 0:
        offset = 0.05
    else:
        offset = offset * 0.05

    for idx, (group, mean_val) in enumerate(means.items()):
        ax.text(
            idx,
            mean_val + offset,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.legend()
    sns.despine()
    st.pyplot(fig)

    # Statistical test
    groups = [group["AIP"].values for _, group in df.groupby(demo_col)]
    if len(groups) == 2:
        stat, pval = stats.ttest_ind(*groups, equal_var=False)
        st.write(f"**T-test p-value:** {pval:.4f}")
    elif len(groups) > 2:
        stat, pval = stats.f_oneway(*groups)
        st.write(f"**ANOVA p-value:** {pval:.4f}")

# --- Underwriting Analysis ---
with uw_tab:
    st.header("Underwriting Disparity Analysis")
    demo_col = st.selectbox("Select demographic group", demographic_cols, key="uw_demo")
    loan_types = uw_df["LoanType"].dropna().unique().tolist()
    selected_loan = st.selectbox("Filter by Loan Type", ["All"] + loan_types, key="uw_loan")
    purpose_filter = st.selectbox("Loan Purpose", ["All"] + uw_df["Purpose"].dropna().unique().tolist())

    df = uw_df.copy()
    if selected_loan != "All":
        df = df[df["LoanType"] == selected_loan]
    if purpose_filter != "All":
        df = df[df["Purpose"] == purpose_filter]

    df = df[df["HmdaActionTaken"].isin(["Loan Originated", "Application denied"])]
    df = df[[demo_col, "HmdaActionTaken"]].dropna()

    st.markdown(f"### Approval Rates by {demo_col}")
    approval_rate = df.groupby([demo_col, "HmdaActionTaken"]).size().unstack().fillna(0)
    st.dataframe(approval_rate)

    # Approval % by group
    approval_rate_percent = approval_rate.div(approval_rate.sum(axis=1), axis=0) * 100

    if "Loan Originated" not in approval_rate_percent.columns:
        st.warning("No 'Loan Originated' records found for this filter combination.")
    else:
        approval_df = approval_rate_percent["Loan Originated"].reset_index()
        approval_df.columns = [demo_col, "Approval %"]
        approval_df = approval_df.sort_values("Approval %", ascending=False)

        overall_approval = (
            approval_rate["Loan Originated"].sum()
            / approval_rate.sum().sum()
            * 100
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(approval_df[demo_col], approval_df["Approval %"], color="#6baed6")

        ax.set_title(f"Approval Rate by {demo_col}", fontsize=14, fontweight="bold")
        ax.set_ylabel("% Approved (Loan Originated)", fontsize=12)
        ax.set_xlabel(demo_col, fontsize=12)
        ax.set_ylim(0, 100)

        # Wrap long x-axis labels
        labels = [str(label) for label in approval_df[demo_col]]
        wrapped_labels = ["\n".join(textwrap.wrap(label, width=10)) for label in labels]
        ax.set_xticklabels(wrapped_labels, rotation=0)

        # Overall approval reference line
        ax.axhline(
            overall_approval,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Overall approval rate ({overall_approval:.1f}%)"
        )

        # Value labels on bars
        for idx, row in approval_df.iterrows():
            ax.text(
                idx,
                row["Approval %"] + 1,
                f"{row['Approval %']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.legend()
        sns.despine()
        st.pyplot(fig)

    # Statistical test
    chi2, pval, _, _ = stats.chi2_contingency(approval_rate.fillna(0))
    st.write(f"**Chi-square test p-value:** `{pval:.4f}`")
