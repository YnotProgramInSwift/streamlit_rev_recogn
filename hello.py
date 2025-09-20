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
planned_revenue_df = pd.read_csv('plan_revenue_data.csv')

def create_projects_df(open_projects_df, actuals_df, budget_df, planned_revenue_df, month_end):
    
    open_projects_df.columns = open_projects_df.columns.str.lower()
    actuals_df.columns = actuals_df.columns.str.lower()
    budget_df.columns = budget_df.columns.str.lower()
    planned_revenue_df.columns = planned_revenue_df.columns.str.lower()

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

    prj_mtd_revenue_df = (
        actuals_df[(actuals_df['account_type'].isin(['Billed Revenue', 'Deferred Revenue'])) & (actuals_df['end_date_of_month'] == month_end)]
        .groupby('project_key')
        .agg({'amount': 'sum'})
        .reset_index()
        .rename(columns={'amount': 'mtd_revenue'})
    )

    prj_mtd_direct_cost_df = (
        actuals_df[(actuals_df['account_type'] == 'Direct Costs') & (actuals_df['end_date_of_month'] == month_end)]
        .groupby('project_key')
        .agg({'amount': 'sum'})
        .reset_index()
        .rename(columns={'amount': 'mtd_direct_cost'})
    )

    prj_mtd_overhead_cost_df = (
        actuals_df[(actuals_df['account_type'] == 'Overheads') & (actuals_df['end_date_of_month'] == month_end)]
        .groupby('project_key')
        .agg({'amount': 'sum'})
        .reset_index()
        .rename(columns={'amount': 'mtd_oh'})
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
            .merge(prj_mtd_revenue_df, on='project_key', how='left')
            .merge(prj_mtd_direct_cost_df, on='project_key', how='left')
            .merge(prj_mtd_overhead_cost_df, on='project_key', how='left')
            .merge(prj_ltd_billed_revenue_df, on='project_key', how='left')
            .merge(prj_ltd_deferred_revenue_df, on='project_key', how='left')
            .merge(prj_ltd_cost_df, on='project_key', how='left')
            .merge(budget_df, on='project_key', how='left')
            .merge(planned_revenue_df, on='project_key', how='left')
            .fillna(0)  # Fill NaN values with 0
        )


    projects_full_df['prj_forecast'] = projects_full_df['prj_budgeted_cost']
    projects_full_df['revenue_adj'] = 0

    projects_full_df['is_capital'] = projects_full_df['functional_area'].str.startswith('L')
    projects_full_df['mtd_cost'] = projects_full_df['mtd_direct_cost'] + projects_full_df['mtd_oh'].where(projects_full_df['is_capital'], 0)
    projects_full_df['ltd_revenue'] = projects_full_df['ltd_billed_revenue'] + projects_full_df['ltd_deferred_revenue']

    projects_full_df['contracted_revenue'] = projects_full_df['ltd_billed_revenue'] #projects_full_df[['ltd_billed_revenue', 'planned_revenue']].min(axis=1)

    projects_full_df['project_is_material'] = projects_full_df['contracted_revenue'] < -10_000

    # drop a specific project code from the DataFrame
    excluded_projects = ['NC-006914','NC-007005']
    projects_full_df = projects_full_df[~projects_full_df['project_code'].isin(excluded_projects)]

    return projects_full_df

def calculate_revenue_adj():
    """Calculate revenue adjustments based on completion percentage and update metrics"""

    project_data_df = st.session_state.project_data_df

    project_data_df['percentage_completion'] = ((project_data_df['ltd_cost'] / project_data_df['prj_forecast'])
                                                    .replace([np.inf, -np.inf], 0)
                                                    .fillna(0)
                                                    .clip(0, 1)
                                                )


    # Business Rule: Any project with less than $10,000 contracted revenue is deemed immaterial, therefore set completion to 100%
    project_data_df['percentage_completion'] = project_data_df['percentage_completion'].where(project_data_df['project_is_material'], 1)

    # Business Rule: If Project budget is less than $10,000 the project has not entered the construction phase, therefore set the completion to 0%
    project_data_df['percentage_completion'] = project_data_df['percentage_completion'].where(project_data_df['prj_budgeted_cost'] >= 10_000, 0)

    # Calculate Revenue Adjustment based on the contracted revenue and the percentage completion
    project_data_df['revenue_adj'] = (
        project_data_df['contracted_revenue'] * project_data_df['percentage_completion'] - (project_data_df['ltd_billed_revenue'] + project_data_df['ltd_deferred_revenue'])
    )

def summary_metrics_component():
    """Component for displaying summary metrics that can be updated independently"""
    display_df = st.session_state.project_data_df

    if st.session_state.type_filter == FilterType.CAPITAL.value:
        display_df = display_df[display_df['is_capital']]
    elif st.session_state.type_filter == FilterType.ACS.value:
        display_df = display_df[~display_df['is_capital']]

    st.header("Monthly Summary")

    total_mtd_revenue_adj = -display_df['revenue_adj'].sum() if st.session_state.show_adjustments else 0
    total_mtd_revenue = -display_df['mtd_revenue'].sum() + total_mtd_revenue_adj
    total_mtd_cost = display_df['mtd_cost'].sum()
    total_mtd_margin = total_mtd_revenue - total_mtd_cost

    col1, col2, col3 = st.columns([3,3,3])

    with col1:
        st.metric(
            label="MTD Revenue",
            value=f"${total_mtd_revenue:,.2f}",
            delta=f"{total_mtd_revenue_adj:,.2f} Adjusted" if total_mtd_revenue_adj != 0 else "0 Adjusted"
        )
    
    with col2:
        st.metric(
            label="MTD Cost",
            value=f"${total_mtd_cost:,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="MTD Margin",
            value=f"${total_mtd_margin:,.2f}",
            delta=f"{(total_mtd_margin/total_mtd_revenue)*100:.0f}%" if total_mtd_revenue > 0 else "0%"
        )
    

def charts_component():
    """Component for displaying charts that can be updated independently"""
    st.header("MTD Revenue vs Costs")

    display_df = st.session_state.project_data_df

    if st.session_state.type_filter == FilterType.CAPITAL.value:
        display_df = display_df[display_df['is_capital']]
    elif st.session_state.type_filter == FilterType.ACS.value:
        display_df = display_df[~display_df['is_capital']]

    # Create data for side-by-side bars
    chart_data = display_df.groupby('functional_area').agg({'mtd_revenue': 'sum', 'revenue_adj': 'sum' , 'mtd_cost': 'sum'}).reset_index()
    chart_data['revenue_adj'] = chart_data['revenue_adj'] if st.session_state.show_adjustments else 0
    chart_data['display_mtd_revenue'] = -(chart_data['mtd_revenue'] + chart_data['revenue_adj'])
    chart_data = chart_data[['functional_area', 'display_mtd_revenue', 'mtd_cost']]
    chart_data = chart_data.melt(id_vars='functional_area', 
                       value_vars=['display_mtd_revenue', 'mtd_cost'],
                       var_name='metric', 
                       value_name='amount')
    
    chart_data['metric'] = chart_data['metric'].map({
        'display_mtd_revenue': 'Revenue',
        'mtd_cost': 'Costs'
    })
    chart_data.rename(columns={'functional_area': 'Functional Area'}, inplace=True)

    st.bar_chart(chart_data, x='Functional Area', y='amount', color='metric', use_container_width=True, stack=False)

@st.fragment
def significant_project_component(project_key):
    """Create a component for displaying and adjusting significant project data"""
    
    # Get project data from session state dataframe
    project_data_df = st.session_state.project_data_df
    project_data = project_data_df[project_data_df['project_key'] == project_key].iloc[0]
    
    # Create a container for the project
    with st.container():

        prj_budgeted_cost = project_data['prj_budgeted_cost']
        #create a new session state variable for this project key
        if f'{project_key}_fct' not in st.session_state:
            st.session_state[f'{project_key}_fct'] = project_data['prj_forecast']

        prj_forecast = st.session_state[f'{project_key}_fct']

        ltd_cost = project_data['ltd_cost']
        percentage_completion = ltd_cost / prj_forecast if prj_forecast != 0 else 0
        percentage_completion = min(max(percentage_completion, 0), 1)
        contracted_revenue = project_data['contracted_revenue']
        revenue_adj = (contracted_revenue * percentage_completion - (project_data['ltd_billed_revenue'] + project_data['ltd_deferred_revenue']))
        
        mtd_cost = project_data['mtd_cost']
        mtd_revenue = project_data['mtd_revenue']

        col1, _, col2, _, col3 = st.columns([9, 1,2,1,2])

        with col1:
            st.markdown(f"#### {project_data['functional_area']}: {project_data['project_code']} - {project_data['project_name']}")
        
        with col2:
            # Forecast input - use current dataframe value
            prj_forecast = st.number_input(
                'Forecast',
                value=float(prj_forecast),
                step=10_000.0,
                format="%.0f",
                key=f'{project_key}_fct',
            )

        with col3:
            if st.button("💾 Save", key=f"commit_{project_key}", type="secondary"):
                # Update the shared dataframe in session state
                project_data_df.loc[project_data_df['project_key'] == project_key, 'prj_forecast'] = prj_forecast
                # Recalculate revenue adjustments for the entire dataframe
                calculate_revenue_adj()
                st.success(f"Forecast of ${prj_forecast:,.0f} saved!")
                # Only rerun this fragment, not the whole app
                st.rerun(scope="fragment")

        # Create 6 columns for metrics
        col1, col2, col3, col4, col5, col6, col7, col8  = st.columns([1, 1, 1, 1, 1, 1, 1, 1])

        with col1:
            st.markdown(f"<div style='font-size: 1em;'><strong>MTD Revenue</strong><br>${-mtd_revenue:,.0f}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div style='font-size: 1em;'><strong>MTD Cost</strong><br>${mtd_cost:,.0f}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div style='font-size: 1em;'><strong>Adjustment</strong><br>${-revenue_adj:,.0f}</div>", unsafe_allow_html=True)

        with col4:
            st.markdown(f"<div style='font-size: 1em;'><strong>Contracted Revenue</strong><br>${-contracted_revenue:,.0f}</div>", unsafe_allow_html=True)

        with col5:
            st.markdown(f"<div style='font-size: 1em;'><strong>LTD Costs</strong><br>${ltd_cost:,.0f}</div>", unsafe_allow_html=True)

        with col6:
            st.markdown(f"<div style='font-size: 1em;'><strong>Budget </strong><br>${prj_budgeted_cost :,.0f}</div>", unsafe_allow_html=True)

        with col7:
            st.markdown(f"<div style='font-size: 1em;'><strong>Forecast </strong><br>${prj_forecast:,.0f}</div>", unsafe_allow_html=True)

        with col8:
            st.markdown(f"<div style='font-size: 1em;'><strong>Completion %</strong><br>{percentage_completion:.2%}</div>", unsafe_allow_html=True)

        # Show adjustment details
        saved_value = st.session_state.get(f"saved_adj_{project_key}", project_data['prj_forecast'])
        if prj_forecast != saved_value:
            st.caption(f"⚠️ Unsaved forecast change: ${prj_forecast - saved_value:,.0f}")

        st.markdown("---")

def main():
    st.set_page_config(
        page_title="Revenue Recognition Dashboard",
        page_icon="",
        layout="wide"
    )
    
    # Set month end date for data filtering
    month_end = pd.Timestamp('2025-07-31')  # Current reporting month
        
    # Initialise the data and store as project_data session_state
    if 'project_data_df' not in st.session_state:
        try:
            st.session_state.project_data_df = create_projects_df(open_projects_df, actuals_df, budget_df, planned_revenue_df, month_end)
            # Calculate initial revenue adjustments
            calculate_revenue_adj()
        except Exception as e:
            st.error(f"Error loading project data: {e}")
            return

    # Initialise session state for showing/hiding adjustments
    if 'show_adjustments' not in st.session_state:
        st.session_state.show_adjustments = False
    
    # Initialise session state for filter type
    if 'type_filter' not in st.session_state:
        st.session_state.type_filter = FilterType.ALL.value

    # Initialise session state for number of top projects
    if 'top_n_projects' not in st.session_state:
        st.session_state.top_n_projects = 10
    
    original_project_df = st.session_state.project_data_df.copy()
    project_data_df = st.session_state.project_data_df

    st.title("Revenue Recognition Dashboard")
    
    # Use the dataframe from session state
    st.markdown("---")

    col1, _, col2, _, col3 = st.columns([3, 1, 3, 1, 3])
    with col1:
        st.toggle("Show Adjustments", key="show_adjustments")
        if st.session_state.show_adjustments:
            st.success("Margin including Revenue Recognition Adjustments")
        else:
            st.info("Margin without Revenue Recognition Adjustments")
    with col2:
        # Filter selectbox
        st.selectbox(
            "Filter for Project Type (Capital/ACS)",
            options=[filter_type.value for filter_type in FilterType],
            key="type_filter"
        )

    with col3:
        st.button("Recalculate Adjustments", type="primary", on_click=calculate_revenue_adj)

    # Summary Section
    summary_metrics_component()

    #st.markdown('---')

    # Bar Chart Section as Fragment
    charts_component()

    ###
    st.markdown('---')
    # st.dataframe(project_data_df, use_container_width=True)
    
    # Top Projects Analysis with Budget Adjustments
    st.header('Top Projects - Analysis & Budget Adjustments')
    st.markdown('The following projects are ranked by their highest MTD Revenue or MTD Cost values. Adjust budgets using the sliders - all calculations will update automatically.')

    # user inputs for top N projects
    st.session_state.top_n_projects = st.selectbox(
        label='Select number of top projects to display',
        options=[10, 15, 20],
    )

    # Sort by either MTD revenue or MTD costs (whichever is higher for each project)
    ranked_projects = original_project_df.copy()

    if st.session_state.type_filter == FilterType.CAPITAL.value:
        ranked_projects = ranked_projects[ranked_projects['is_capital']]
    elif st.session_state.type_filter == FilterType.ACS.value:
        ranked_projects = ranked_projects[~ranked_projects['is_capital']]

    ranked_projects['display_revenue'] = (ranked_projects['mtd_revenue'] + ranked_projects['revenue_adj']).abs()
    ranked_projects['max_mtd_value'] = ranked_projects[['display_revenue', 'mtd_cost']].max(axis=1)
    ranked_projects = ranked_projects.nlargest(st.session_state.top_n_projects, 'max_mtd_value')

    # Display each project using the component
    for index, project_row in ranked_projects.iterrows():
        significant_project_component(project_row['project_key'])



    st.markdown('---')

    # Raw Data Section (expandable)
    with st.expander('📋 View Raw Data'):
        st.dataframe(project_data_df, use_container_width=True)
    

if __name__ == '__main__':
    main()
