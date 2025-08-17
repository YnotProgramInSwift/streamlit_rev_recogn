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

    excluded_projects = ['NC-006914']

    projects_full_df = projects_full_df[~projects_full_df['project_code'].isin(excluded_projects)]

    return projects_full_df

def calculate_metrics(df):
    """Calculate revenue adjustments based on completion percentage and update metrics"""
    df_calc = df.copy()

    df_calc['budget_adj'] = 0
    df_calc['revenue_adj'] = 0
    # drop a specific project code from the DataFrame

    # mtd_cost = direct costs + overheads if the functional_area starts with 'L' otherwise just direct costs
    df_calc['is_capital'] = df_calc['functional_area'].str.startswith('L')
    df_calc['mtd_cost'] = df_calc['mtd_direct_cost'] + df_calc['mtd_oh'].where(df_calc['is_capital'], 0)
    df_calc['ltd_revenue'] = df_calc['ltd_billed_revenue'] + df_calc['ltd_deferred_revenue']

    # create contracted_revenue = min of ltd_billed_revenue and planned_revenue
    df_calc['contracted_revenue'] = df_calc[['ltd_billed_revenue', 'planned_revenue']].min(axis=1)

    df_calc['project_is_material'] = df_calc['contracted_revenue'] < -10_000

    df_calc['percentage_completion'] = df['ltd_cost'] / (df_calc['prj_budgeted_cost'] + df_calc['budget_adj'])

    # Replace NaN, inf, and -inf with 0, then clip between 0 and 1
    df_calc['percentage_completion'] = (df_calc['percentage_completion']
                                        .replace([np.inf, -np.inf], 0)
                                        .fillna(0)
                                        .clip(0, 1)
                                        )

    df_calc['percentage_completion'] = df_calc['percentage_completion'].where(df_calc['project_is_material'], 1)

    df_calc['revenue_adj'] = (
        df_calc['contracted_revenue'] * df_calc['percentage_completion'] - (df_calc['ltd_billed_revenue'] + df_calc['ltd_deferred_revenue'])
    )

    return df_calc

def significant_project_component(project_data, df_original, row_index):
    """Create a component for displaying and adjusting significant project data"""
    
    # Create a container for the project
    with st.container():
        st.markdown(f"### {project_data['functional_area']}: {project_data['project_code']} - {project_data['project_name']}")
        
        project_key = project_data['project_key']
        budget_adj = st.session_state[f'adj_{project_key}'] if f'adj_{project_key}' in st.session_state else project_data['budget_adj']
        prj_budgeted_cost = project_data['prj_budgeted_cost']
        ltd_cost = project_data['ltd_cost']
        percentage_completion = ltd_cost / (prj_budgeted_cost + budget_adj) if (prj_budgeted_cost + budget_adj) != 0 else 0
        percentage_completion = min(max(percentage_completion, 0), 1)
        ltd_revenue = -project_data['contracted_revenue'] * percentage_completion



        # Create 6 columns for metrics
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"<div style='font-size: 1em;'><strong>LTD Revenue</strong><br>${ltd_revenue:,.0f}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div style='font-size: 1em;'><strong>LTD Costs</strong><br>${ltd_cost:,.0f}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div style='font-size: 1em;'><strong>LTD Budget (incl Adj)</strong><br>${(prj_budgeted_cost + budget_adj):,.0f}</div>", unsafe_allow_html=True)

        with col4:
            st.markdown(f"<div style='font-size: 1em;'><strong>Completion %</strong><br>{percentage_completion:.2%}</div>", unsafe_allow_html=True)

        with col5:
            # Numerical input field for revenue adjustment
            adjustment = st.number_input(
                "Revenue Adj",
                value=float(project_data['budget_adj']),
                step=1000.0,
                format="%.0f",
                key=f"adj_{project_key}",
                label_visibility="visible"
            )

        with col6:
            # Commit button
            if st.button("💾 Commit", key=f"commit_{project_key}", type="secondary"):
                # Update the original dataframe
                df_original.loc[row_index, 'revenue_adj'] = adjustment
                st.success(f"Revenue adjustment of ${adjustment:,.0f} committed for {project_data['project_code']}")
                st.rerun()
        
        # Show adjustment details
        if adjustment != project_data['revenue_adj']:
            st.caption(f"⚠️ Uncommitted change: ${adjustment - project_data['revenue_adj']:,.0f}")
        elif adjustment != 0:
            st.caption(f"✅ Current adjustment: ${adjustment:,.0f}")
        
        st.markdown("---")


def main():
    st.set_page_config(
        page_title="Revenue Recognition Dashboard",
        page_icon="",
        layout="wide"
    )
    
    # Initialise session state for showing/hiding adjustments
    if 'show_adjustments' not in st.session_state:
        st.session_state.show_adjustments = False
    
    # Initialise session state for filter type
    if 'type_filter' not in st.session_state:
        st.session_state.type_filter = FilterType.ALL.value

    # Initialise session state for number of top projects
    if 'top_n_projects' not in st.session_state:
        st.session_state.top_n_projects = 10

    st.title("Revenue Recognition Dashboard")
      
    # Set month end date for data filtering
    month_end = pd.Timestamp('2025-07-31')  # Current reporting month
    
    # Create projects dataframe using the new function
    try:
        df = create_projects_df(open_projects_df, actuals_df, budget_df, planned_revenue_df, month_end)
        st.success(f"Loaded {len(df)} projects for analysis")
    except Exception as e:
        st.error(f"Error loading project data: {e}")
        return
    st.markdown("---")
    
    # Always show original data first with basic calculations
    df_calc = calculate_metrics(df)

    col1, buffer, col2 = st.columns([3, 1, 5])
    with col1:
        st.toggle("Show Adjustments", key="show_adjustments")
        if st.session_state.show_adjustments:
            st.success("Margin including Revenue Recognition Adjustments")
        else:
            st.info("Margin without Revenue Recognition Adjustments")
    with col2:
        # Filter selectbox
        filter_selection = st.selectbox(
            "Filter Projects",
            options=[filter_type.value for filter_type in FilterType],
            key="filter_selectbox"
        )
        st.session_state.type_filter = filter_selection
    # Summary Section

    display_df = df_calc.copy()
    if st.session_state.type_filter == FilterType.CAPITAL.value:
        display_df = display_df[display_df['is_capital']]
    elif st.session_state.type_filter == FilterType.ACS.value:
        display_df = display_df[~display_df['is_capital']] 

    st.header("Monthly Summary")

    total_mtd_revenue_adj = -display_df['revenue_adj'].sum() if st.session_state.show_adjustments else 0
    total_mtd_revenue = -display_df['mtd_revenue'].sum() + total_mtd_revenue_adj
    total_mtd_cost = display_df['mtd_cost'].sum()
    total_mtd_margin = total_mtd_revenue - total_mtd_cost
    
    col1, col2, col3 = st.columns(3)
    
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
    
    st.markdown("---")
    
    # Bar Chart Section
    st.header("MTD Revenue vs Costs")
    
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

    ###
    st.markdown("---")
    # st.dataframe(df_calc, use_container_width=True)
    
    # Top Projects Analysis with Budget Adjustments
    st.header("Top Projects - Analysis & Budget Adjustments")
    st.markdown("The following projects are ranked by their highest MTD Revenue or MTD Cost values. Adjust budgets using the sliders - all calculations will update automatically.")

    # user inputs for top N projects
    st.session_state.top_n_projects = st.selectbox(
        label="Select number of top projects to display",
        options=[10, 15, 20, 25],
    )

    # Sort by either MTD revenue or MTD costs (whichever is higher for each project)
    ranked_projects = display_df.copy()
    ranked_projects['display_revenue'] = (ranked_projects['mtd_revenue'] + ranked_projects['revenue_adj']).abs()
    ranked_projects['max_mtd_value'] = ranked_projects[['display_revenue', 'mtd_cost']].max(axis=1)
    ranked_projects = ranked_projects.nlargest(st.session_state.top_n_projects, 'max_mtd_value')

    # Display each project using the component
    for index, project_row in ranked_projects.iterrows():
        row_index = df_calc.index[df_calc['project_key'] == project_row['project_key']].tolist()[0]
        significant_project_component(project_row, df_calc, row_index)


    
    st.markdown("---")
    
    # Raw Data Section (expandable)
    with st.expander("📋 View Raw Data"):
        st.dataframe(df_calc, use_container_width=True)
    

if __name__ == "__main__":
    main()
