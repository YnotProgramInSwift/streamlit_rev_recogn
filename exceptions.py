import streamlit as st
import pandas as pd
import numpy as np
from enum import Enum
from datetime import date, timedelta

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

    prj_ltd_billed_revenue_df = (
        actuals_df[(actuals_df['account_type'] == 'Billed Revenue') & (actuals_df['end_date_of_month'] <= month_end)]
        .groupby('project_key')
        .agg({'amount': 'sum'})
        .reset_index()
        .rename(columns={'amount': 'ltd_billed_revenue'})
    )
                        
    prj_ltd_deferred_revenue_df = (
        actuals_df[(actuals_df['account_type'] == 'Deferred Revenue') & (actuals_df['end_date_of_month'] <= month_end)]
        .groupby('project_key')
        .agg({'amount': 'sum'})
        .reset_index()
        .rename(columns={'amount': 'ltd_deferred_revenue'})
    )

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
            .merge(prj_ltd_billed_revenue_df, on='project_key', how='left')
            .merge(prj_ltd_deferred_revenue_df, on='project_key', how='left')
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

def display_budget_exceptions(df):

    functional_areas_list = ['All']
    project_managers_list = ['All']
    cost_thresholds_dict = {"$50,000": 50_000, "$100,000": 100_000, "$150,000": 150_000}
    budget_exception_rates_dict = {"10%": 0.1, "20%": 0.2, "30%": 0.3, "40%": 0.4, "50%": 0.5, "60%": 0.6, "70%": 0.7, "80%": 0.8}

# Initialise session state for filter type
    if 'manager' not in st.session_state:
        st.session_state.manager = 'All'

    if 'functional_area' not in st.session_state:
        st.session_state.functional_area = 'All'

    if 'project_cost_threshold' not in st.session_state:
        st.session_state.project_cost_threshold = list(cost_thresholds_dict.keys())[0]

    if 'budget_exception_rate' not in st.session_state:
        st.session_state.budget_exception_rate = list(budget_exception_rates_dict.keys())[2]


    cost_thresholds_value = cost_thresholds_dict[st.session_state.project_cost_threshold]
    budget_exception_rate_value = budget_exception_rates_dict[st.session_state.budget_exception_rate]
    
    display_df = calculate_exceptions(df.copy(), cost_threshold=cost_thresholds_value, budget_threshold=budget_exception_rate_value)
    
    functional_areas_list.extend(display_df['functional_area'].unique().tolist())
    project_managers_list.extend(display_df['project_manager'].unique().tolist())

    if st.session_state.manager != 'All':
        display_df = display_df[display_df['project_manager'] == st.session_state.manager]
    
    if st.session_state.functional_area != 'All':
        display_df = display_df[display_df['functional_area'] == st.session_state.functional_area]

    subtitle = f'There are {len(display_df)} projects where life-to-date costs exceeds {st.session_state.project_cost_threshold}, and the budget is less than {st.session_state.budget_exception_rate} of the spent.'


    st.subheader('Budget Exceptions')
    st.warning(subtitle)

    display_df = display_df[['project_code', 'project_name', 'project_manager', 'functional_area', 'ltd_billed_revenue', 'ltd_cost', 'prj_budgeted_cost']].copy()
    #format numbers
    display_df['ltd_billed_revenue'] = -display_df['ltd_billed_revenue']
    display_df[['ltd_billed_revenue', 'ltd_cost', 'prj_budgeted_cost']] = display_df[['ltd_billed_revenue', 'ltd_cost', 'prj_budgeted_cost']].map(lambda x: f"${x:,.0f}")

    display_df.rename(
        columns={
            'project_code': 'Project Code',
            'project_name': 'Project Name',
            'project_manager': 'Project Manager',
            'functional_area': 'Functional Area',
            'ltd_billed_revenue': 'Total Billed Revenue',
            'ltd_cost': 'Total Costs',
            'prj_budgeted_cost': 'Budgeted Costs in SAP'
        },
        inplace=True
    )

    col1, _, col2, _, col3, _, col4 = st.columns([3, 1, 3, 1, 3, 1, 3])

    with col1:
        st.selectbox("Select Project Manager", options=project_managers_list, key="manager")

    with col2:
        st.selectbox("Select Functional Area", options=functional_areas_list, key="functional_area")
    with col3:
        #change display values in cost_thresholds_list to $
        st.selectbox("Select Cost Threshold", options=cost_thresholds_dict.keys(), key="project_cost_threshold")

    with col4:
        st.selectbox("Select Budget Exception Rate", options=budget_exception_rates_dict.keys(), key="budget_exception_rate")

    # Raw Data Section (expandable)
    st.dataframe(display_df, use_container_width=True)

def display_inactive_projects(df):
    
    display_df = df.copy()
    display_df['last_transaction_month'] = pd.to_datetime(display_df['last_transaction_month'], errors='coerce')
    display_df['last_transaction_month'] = display_df['last_transaction_month'].dt.strftime('%Y-%m')
    display_df['ltd_deferred_revenue'] = round(display_df['ltd_deferred_revenue'], 0)
    display_df = display_df[display_df['ltd_deferred_revenue'] != 0]
    display_df['prj_full_name'] = display_df['project_code'] + ' - ' + display_df['project_name']
    display_df = display_df.groupby(['prj_full_name', 'last_transaction_month', 'functional_area', 'project_manager']).agg({'ltd_deferred_revenue': 'sum'}).reset_index()
    display_df['is_open'] = True

    display_df.rename(
        columns={
            'prj_full_name': 'Project Name',
            'last_transaction_month': 'Last Transaction Month',
            'ltd_deferred_revenue': 'Remaining Unrecognised Revenue',
            'functional_area': 'Functional Area',
            'project_manager': 'Project Manager'
        }, inplace=True)

    st.subheader("Unrecognised Revenue by last transaction month")

    st.slider(
        "Select the range to filter for the years the transactions last occurred", 
        2020, 2025, (2020, 2025),
        key="year_range"
    )
    min_year, max_year = st.session_state.year_range

    display_df = display_df[display_df['Last Transaction Month'].between(f"{min_year}-01", f"{max_year}-12")]

    remaining_revenue = display_df['Remaining Unrecognised Revenue'].sum()
    projects_count = display_df['Project Name'].nunique()

    st.warning(
        f'''There are {projects_count} open projects with total unrecognised revenue balance of ${remaining_revenue:,.0f} 
        where the last transaction occurred between {min_year} and {max_year}''')

    st.bar_chart(
        data=display_df,
        x='Last Transaction Month',
        y='Remaining Unrecognised Revenue',
        color='Project Name',
        height=400,
    )

    with st.expander("View Projects Data", expanded=False):
        st.dataframe(display_df, use_container_width=True)

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

    tab1, tab2 = st.tabs(["Budget Exceptions", "Inactive Projects with Deferred Revenue "])

    with tab1:
        display_budget_exceptions(df)

    with tab2:
        display_inactive_projects(df)

    st.markdown("---")

    
    

if __name__ == "__main__":
    main()
