import streamlit as st
import pandas as pd
import numpy as np
from enum import Enum

class FilterType(Enum):
    ALL = "All"
    CAPITAL = "Capital"
    ACS = "ACS"

open_projects_df = pd.read_csv('open_projects_listing.csv')
actuals_df = pd.read_csv('actuals_data.csv')
budget_df = pd.read_csv('budget_data.csv')

pd.options.display.float_format = '{:,.2f}'.format


def create_projects_df(open_projects_df, actuals_df, budget_df, month_end):

    open_projects_df.columns = open_projects_df.columns.str.lower()
    actuals_df.columns = actuals_df.columns.str.lower()
    budget_df.columns = budget_df.columns.str.lower()

    actuals_df['end_date_of_month'] = pd.to_datetime(actuals_df['end_date_of_month'], dayfirst=True,  errors='coerce')

    prj_ltd_cost_df = (
        actuals_df[(~actuals_df['account_type'].str.contains('Revenue')) & (actuals_df['end_date_of_month'] <= month_end)]
        .groupby('project_key')
        .agg({'amount': 'sum'})
        .reset_index()
        .rename(columns={'amount': 'ltd_cost'})
    )

    prj_last_transac_mth_df = (
        actuals_df.groupby('project_key')
        .agg({'end_date_of_month': 'max'})
        .reset_index()
        .rename(columns={'end_date_of_month': 'last_transaction_month'})
    )

    projects_full_df = (
        open_projects_df[['project_key', 'project_code', 'project_name', 'project_manager', 'functional_area']]
            .merge(prj_last_transac_mth_df, on='project_key', how='left')
            .merge(prj_ltd_cost_df, on='project_key', how='left')
            .merge(budget_df, on='project_key', how='left')
            .fillna(0)  # Fill NaN values with 0
        )

    excluded_projects = ['NC-006914']

    projects_full_df = projects_full_df[~projects_full_df['project_code'].isin(excluded_projects)]

    return projects_full_df

def calculate_exceptions(df, cost_threshold = 100_000, budget_threshold = 0.3):
    # Identify projects with revenue below the threshold
    df['budget_error'] = ((df['ltd_cost'] > cost_threshold) & (df['prj_budgeted_cost'] / df['ltd_cost'] < budget_threshold))
    return df[df['budget_error']]

def display_budget_exceptions(subtitle, budget_exceptions_df, project_managers_list, functional_areas_list):

    st.subheader('Budget Exceptions')
    st.success(subtitle)

    display_df = budget_exceptions_df[['project_code', 'project_name', 'project_manager', 'functional_area', 'ltd_cost', 'prj_budgeted_cost']].copy()
    #format numbers
    display_df[['ltd_cost', 'prj_budgeted_cost']] = display_df[['ltd_cost', 'prj_budgeted_cost']].applymap(lambda x: f"${x:,.0f}")

    display_df.rename(
        columns={
            'project_code': 'Project Code',
            'project_name': 'Project Name',
            'project_manager': 'Project Manager',
            'functional_area': 'Functional Area',
            'ltd_cost': 'Life-to-Date Costs',
            'prj_budgeted_cost': 'Budgeted Costs'
        },
        inplace=True
    )

    # Initialise session state for filter type
    if 'manager' not in st.session_state:
        st.session_state.manager = 'All'

    if 'functional_area' not in st.session_state:
        st.session_state.functional_area = 'All'
    

    col1, _, col2 = st.columns([5, 1, 5])

    with col1:
        st.selectbox("Select Project Manager", options=project_managers_list, key="manager")

    with col2:
        st.selectbox("Select Functional Area", options=functional_areas_list, key="functional_area")

    if st.session_state.manager != 'All':
        display_df = display_df[display_df['project_manager'] == st.session_state.manager]

    if st.session_state.functional_area != 'All':
        display_df = display_df[display_df['functional_area'] == st.session_state.functional_area]

    # Raw Data Section (expandable)
    st.dataframe(display_df, use_container_width=True)

def display_inactive_projects(df):
    return 


def main():
    st.set_page_config(
        page_title="Exceptions Dashboard",
        page_icon="",
        layout="wide"
    )


    st.title("Projects Exceptions Dashboard")

    # Set month end date for data filtering
    month_end = pd.Timestamp('2025-07-31')  # Current reporting month
     # Create projects dataframe using the new function
    try:
        df = create_projects_df(open_projects_df, actuals_df, budget_df, month_end)
        st.success(f"Loaded {len(df)} projects for analysis")
    except Exception as e:
        st.error(f"Error loading project data: {e}")
        return
    st.markdown("---")

    #data for budget exceptions
    cost_threshold = 50_000
    budget_exception_rate = 0.5
    budget_exceptions_df = calculate_exceptions(df=df, cost_threshold=cost_threshold, budget_threshold=budget_exception_rate)
    subtitle = f'List of projects where life-to-date costs exceeds ${cost_threshold:,.0f}, and the budget is less than {budget_exception_rate*100:.0f}%'
    functional_areas_list = budget_exceptions_df['functional_area'].unique().tolist() + ['All']
    project_managers_list = budget_exceptions_df['project_manager'].unique().tolist() + ['All'] 

    tab1, tab2 = st.tabs(["Budget Exceptions", "Inactive Projects with Deferred Revenue "])

    with tab1:
        display_budget_exceptions(subtitle, budget_exceptions_df, project_managers_list, functional_areas_list)

    with tab2:
        display_inactive_projects(df)

    st.markdown("---")

    
    

if __name__ == "__main__":
    main()
