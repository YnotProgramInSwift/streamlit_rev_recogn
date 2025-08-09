import streamlit as st
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample dataframe with revenue recognition data"""
    np.random.seed(42)  # For reproducible data
    
    data = {
        'project_id': [f'PRJ-{i:03d}' for i in range(1, 16)],
        'mtd_billed_revenue': np.random.uniform(10000, 50000, 15).round(2),
        'mtd_revenue_adj': np.zeros(15),
        'mtd_cost': np.random.uniform(8000, 35000, 15).round(2),
        'ltd_billed_revenue': np.random.uniform(50000, 200000, 15).round(2),
        'ltd_deferred_revenue': -np.random.uniform(20000, 100000, 15).round(2),
        'ltd_costs': np.random.uniform(40000, 150000, 15).round(2),
        'ltd_budget': np.random.uniform(80000, 250000, 15).round(2),
    }
    
    return pd.DataFrame(data)

def calculate_metrics(df, budget_adjustments):
    """Calculate derived metrics for the dataframe"""
    df_calc = df.copy()
    
    # Apply budget adjustments
    for project_id, adjustment in budget_adjustments.items():
        df_calc.loc[df_calc['project_id'] == project_id, 'ltd_budget'] += adjustment
    
    # Calculate derived columns
    df_calc['ltd_revenue'] = df_calc['ltd_billed_revenue'] + df_calc['ltd_deferred_revenue']
    df_calc['percentage_completion'] = (df_calc['ltd_costs'] / df_calc['ltd_budget']).clip(0, 1)
    
    # Calculate adjusted deferred revenue based on completion percentage
    df_calc['mtd_revenue_adj'] = (df_calc['ltd_billed_revenue'] * df_calc['percentage_completion'] 
                                          - (df_calc['ltd_billed_revenue'] + df_calc['ltd_deferred_revenue']))
    
    df_calc['mtd_revenue'] = df_calc['mtd_billed_revenue'] + df_calc['mtd_revenue_adj']
    df_calc['mtd_margin'] = df_calc['mtd_revenue'] - df_calc['mtd_cost']
    df_calc['ltd_margin'] = df_calc['ltd_revenue'] - df_calc['ltd_costs']


    return df_calc

def significant_project_component(project_data, original_budget, project_id, df_original):
    """Create a component for displaying and adjusting significant project data"""
    
    # Create a container for the project
    with st.container():
        st.markdown(f"### {project_id}")
        
        # Create 4 columns for metrics and 2 columns for adjustment
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1.5, 1])
        
        with col1:
            st.metric("LTD Revenue", f"${project_data['ltd_revenue']:,.0f}")
        
        with col2:
            st.metric("LTD Costs", f"${project_data['ltd_costs']:,.0f}")
        
        with col3:
            st.metric("LTD Budget", f"${project_data['ltd_budget']:,.0f}")
        
        with col4:
            st.metric("Completion %", f"{project_data['percentage_completion']:.1%}")
        
        with col5:
            # Budget adjustment slider
            current_adjustment = st.session_state.budget_adjustments.get(project_id, 0)
            adjustment = st.slider(
                "Budget Adjustment",
                min_value=-original_budget * 0.5,
                max_value=original_budget * 0.5,
                value=float(current_adjustment),
                step=1000.0,
                format="$%.0f",
                key=f"adj_{project_id}",
                label_visibility="collapsed"
            )
            
            # Check if adjustment changed and trigger immediate recalculation
            if adjustment != current_adjustment:
                st.session_state.budget_adjustments[project_id] = adjustment
                # Automatically recalculate when slider changes
                st.session_state.calculated_data = calculate_metrics(df_original, st.session_state.budget_adjustments)
                st.rerun()
            
            # Update session state
            st.session_state.budget_adjustments[project_id] = adjustment
            
            # Show adjustment details
            new_budget = original_budget + adjustment
            if adjustment != 0:
                st.caption(f"Adjustment: ${adjustment:,.0f}")
                st.caption(f"New Budget: ${new_budget:,.0f}")
            else:
                st.caption("No adjustment")
        
        with col6:
            st.metric("MTD Adj", f"${project_data['mtd_revenue_adj']:,.0f}")
        
        st.markdown("---")

def main():
    st.set_page_config(
        page_title="Revenue Recognition Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Revenue Recognition Dashboard")
    st.markdown("---")
    
    # Initialize session state for budget adjustments
    if 'budget_adjustments' not in st.session_state:
        st.session_state.budget_adjustments = {}
    
    if 'calculated_data' not in st.session_state:
        st.session_state.calculated_data = None
    
    # Create sample data
    df = create_sample_data()
    
    # Always show original data first
    df_display = df.copy()
    df_display['mtd_revenue'] = df_display['mtd_billed_revenue'] + df_display['mtd_revenue_adj']
    df_display['mtd_margin'] = df_display['mtd_revenue'] - df_display['mtd_cost']
    df_display['ltd_revenue'] = df_display['ltd_billed_revenue'] + df_display['ltd_deferred_revenue']
    df_display['percentage_completion'] = (df_display['ltd_costs'] / df_display['ltd_budget']).clip(0, 1)
    
    # Calculate button
    st.header("🔄 Calculate Metrics")
    if st.button("📊 Calculate", type="primary"):
        # Calculate metrics with current adjustments and store in session state
        st.session_state.calculated_data = calculate_metrics(df, st.session_state.budget_adjustments)
    
    # Use calculated data if available, otherwise use original data
    if st.session_state.calculated_data is not None:
        df_calc = st.session_state.calculated_data
        st.success("Showing calculated results with budget adjustments applied.")
    else:
        df_calc = df_display
        st.info("Showing original data. Use sliders to adjust budgets - calculations will update automatically.")
    
    # Summary Section
    st.header("📈 Monthly Summary")
    
    total_mtd_revenue = df_calc['mtd_revenue'].sum()
    total_mtd_revenue_adj = df_calc['mtd_revenue_adj'].sum()
    total_mtd_cost = df_calc['mtd_cost'].sum()
    total_mtd_margin = total_mtd_revenue - total_mtd_cost
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="MTD Revenue",
            value=f"${total_mtd_revenue:,.2f}",
            delta=f"Adjustment: ${total_mtd_revenue_adj:,.2f}" if total_mtd_revenue_adj != 0 else "Adjustment: $0"
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
            delta=f"{(total_mtd_margin/total_mtd_revenue)*100:.1f}%" if total_mtd_revenue > 0 else "0%"
        )
    
    st.markdown("---")
    
    # Bar Chart Section
    st.header("📊 MTD Revenue vs Costs")
    
    # Create data for side-by-side bars
    chart_data = pd.DataFrame({
        'Revenue': [total_mtd_revenue],
        'Cost': [total_mtd_cost]
    })
    
    st.bar_chart(chart_data, height=400, stack=False )
    
    st.markdown("---")
    
    # Top 10 Projects Analysis with Budget Adjustments
    st.header("🎯 Top 10 Projects - Analysis & Budget Adjustments")
    st.markdown("The following projects are ranked by their highest MTD Revenue or MTD Cost values. Adjust budgets using the sliders - all calculations will update automatically.")
    
    # Sort by either MTD revenue or MTD costs (whichever is higher for each project)
    df_calc['max_mtd_value'] = df_calc[['mtd_revenue', 'mtd_cost']].max(axis=1)
    top_10_df = df_calc.nlargest(10, 'max_mtd_value').copy()
    
    # Display each project using the component
    for _, project_row in top_10_df.iterrows():
        project_id = project_row['project_id']
        original_budget = df[df['project_id'] == project_id]['ltd_budget'].iloc[0]
        significant_project_component(project_row, original_budget, project_id, df)
    
    # Reset button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Reset All Adjustments", use_container_width=True):
            st.session_state.budget_adjustments = {}
            st.session_state.calculated_data = None
            st.rerun()
    
    st.markdown("---")
    
    # Raw Data Section (expandable)
    with st.expander("📋 View Raw Data"):
        st.dataframe(df_calc, use_container_width=True)

if __name__ == "__main__":
    main()
