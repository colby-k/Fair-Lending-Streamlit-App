import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Page setup
st.set_page_config(page_title="Fair Lending Analysis", layout="wide")
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

    # Visualization (customized to match notebook style)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=demo_col, y="AIP", data=df, ax=ax, palette="Set2", linewidth=1.2, fliersize=3)
    ax.set_title(f"AIP Distribution by {demo_col}", fontsize=14, fontweight='bold')
    ax.set_xlabel(demo_col, fontsize=12)
    ax.set_ylabel("AIP", fontsize=12)
    plt.xticks(rotation=30)
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

    # Bar chart (customized)
    approval_rate_percent = approval_rate.div(approval_rate.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    approval_rate_percent.plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
    ax.set_title(f"Approval vs Denial % by {demo_col}", fontsize=14, fontweight='bold')
    ax.set_ylabel("% of Applications", fontsize=12)
    ax.set_xlabel(demo_col, fontsize=12)
    plt.xticks(rotation=30)
    sns.despine()
    st.pyplot(fig)

    # Statistical test (Chi-square)
    chi2, pval, _, _ = stats.chi2_contingency(approval_rate.fillna(0))
    st.write(f"**Chi-square test p-value:** {pval:.4f}")
