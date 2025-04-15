import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from bioinfokit.analys import stat
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fair Lending Analysis", layout="wide")
st.title("Fair Lending Analysis Dashboard")

# Tabs for the two analyses
tab1, tab2 = st.tabs(["🏷️ Price Testing", "📝 Credit Decision Testing"])

# --- PRICE TESTING TAB ---
with tab1:
    st.header("Price Testing")

    pricing_file = st.file_uploader("Upload Pricing Data", type=["csv", "xlsx"], key="price")
    if pricing_file:
        df_price = pd.read_csv(pricing_file) if pricing_file.name.endswith(".csv") else pd.read_excel(pricing_file)
        st.write("### Data Preview")
        st.dataframe(df_price.head())

        # Let user select group and numeric fields
        group_field = st.selectbox("Select Grouping Column (e.g. Race, Sex)", df_price.columns)
        value_field = st.selectbox("Select Numeric Column (e.g. Rate, APR)", df_price.select_dtypes(include=np.number).columns)

        st.write(f"### Distribution of {value_field} by {group_field}")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_price, x=group_field, y=value_field, ax=ax)
        st.pyplot(fig)

        st.write("### ANOVA Test Results")
        res = stat()
        res.anova_stat(df=df_price, res_var=value_field, factor_var=group_field)
        st.dataframe(res.anova_summary.round(4))

# --- CREDIT DECISION TESTING TAB ---
with tab2:
    st.header("Credit Decision Testing")

    uw_file = st.file_uploader("Upload Underwriting Data", type=["csv", "xlsx"], key="uw")
    if uw_file:
        df_uw = pd.read_csv(uw_file) if uw_file.name.endswith(".csv") else pd.read_excel(uw_file)
        st.write("### Data Preview")
        st.dataframe(df_uw.head())

        # Let user select group and outcome fields
        group_field_uw = st.selectbox("Select Grouping Column (e.g. Race, Sex)", df_uw.columns, key="group_uw")
        outcome_field_uw = st.selectbox("Select Outcome Column (e.g. Approved, Denied)", df_uw.columns, key="outcome_uw")

        st.write(f"### Approval Rates by {group_field_uw}")
        summary = df_uw.groupby(group_field_uw)[outcome_field_uw].value_counts(normalize=True).unstack().fillna(0)
        st.dataframe(summary.round(3))

        st.write("### Chi-Square Test")
        contingency = pd.crosstab(df_uw[group_field_uw], df_uw[outcome_field_uw])
        chi2, p, dof, ex = stats.chi2_contingency(contingency)
        st.write(f"Chi-square Statistic: {chi2:.4f}")
        st.write(f"p-value: {p:.4f}")

        fig2, ax2 = plt.subplots()
        summary.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_ylabel("Proportion")
        st.pyplot(fig2)
