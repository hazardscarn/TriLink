import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import numpy as np
import pandas as pd
from pprint import pprint
warnings.filterwarnings('ignore')


def create_churn_correlation_plot(df, churn_column='mobile_churn', figsize=(12, 8)):
    """
    Create a comprehensive correlation plot for churn analysis
    
    Parameters:
    df: DataFrame with service data
    churn_column: Name of the churn column
    figsize: Figure size tuple
    """
    
    # Make a copy to avoid modifying original data
    analysis_df = df.copy()
    
    # Prepare data for correlation analysis
    analysis_df = prepare_correlation_data(analysis_df)
    
    # Select numeric columns for correlation
    numeric_columns = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns and other non-meaningful columns
    exclude_cols = [col for col in numeric_columns if 'id' in col.lower() or 'counter' in col.lower()]
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    # Ensure churn column is included
    if churn_column not in numeric_columns:
        numeric_columns.append(churn_column)
    
    # Calculate correlation matrix
    correlation_matrix = analysis_df[numeric_columns].corr()
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(f'Correlation Matrix - Focus on {churn_column.replace("_", " ").title()}', 
              fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Return correlation with churn column
    churn_correlations = correlation_matrix[churn_column].drop(churn_column)
    return churn_correlations.sort_values(key=abs, ascending=False)

def prepare_correlation_data(df):
    """
    Prepare data by encoding categorical variables and creating useful features
    """
    analysis_df = df.copy()
    
    # Encode categorical variables
    categorical_columns = ['plan_type', 'contract_type']
    le = LabelEncoder()
    
    for col in categorical_columns:
        if col in analysis_df.columns:
            analysis_df[f'{col}_encoded'] = le.fit_transform(analysis_df[col].astype(str))
    
    # Create binary features from categorical data
    if 'plan_type' in analysis_df.columns:
        analysis_df['is_unlimited'] = analysis_df['plan_type'].str.contains('Unlimited', na=False).astype(int)
        analysis_df['is_premium'] = analysis_df['plan_type'].str.contains('Premium', na=False).astype(int)
        analysis_df['is_limited'] = analysis_df['plan_type'].str.contains('Limited', na=False).astype(int)
    
    if 'contract_type' in analysis_df.columns:
        analysis_df['is_month_to_month'] = (analysis_df['contract_type'] == 'Month_to_Month').astype(int)
        analysis_df['is_long_contract'] = analysis_df['contract_type'].isin(['24_Month', '36_Month']).astype(int)
    
    # Create derived features
    if 'monthly_cost' in analysis_df.columns and 'line_count' in analysis_df.columns:
        analysis_df['cost_per_line'] = analysis_df['monthly_cost'] / analysis_df['line_count']
    
    if 'data_overage_frequency' in analysis_df.columns:
        analysis_df['high_overage'] = (analysis_df['data_overage_frequency'] > 2).astype(int)
        analysis_df['any_overage'] = (analysis_df['data_overage_frequency'] > 0).astype(int)
    
    # Handle boolean columns
    bool_columns = analysis_df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        analysis_df[col] = analysis_df[col].astype(int)
    
    return analysis_df

def create_churn_bar_plot(df, churn_column='mobile_churn', top_n=15, figsize=(12, 8)):
    """
    Create a horizontal bar plot showing correlations with churn
    """
    analysis_df = prepare_correlation_data(df)
    
    # Get numeric columns
    numeric_columns = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [col for col in numeric_columns if 'id' in col.lower()]
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    # Calculate correlations
    correlations = analysis_df[numeric_columns].corrwith(analysis_df[churn_column])
    correlations = correlations.drop(churn_column).sort_values(key=abs, ascending=False)
    
    # Take top N correlations
    top_correlations = correlations.head(top_n)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create color map based on positive/negative correlation
    colors = ['red' if x > 0 else 'blue' for x in top_correlations.values]
    
    # Horizontal bar plot
    bars = plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
    
    # Customize the plot
    plt.yticks(range(len(top_correlations)), top_correlations.index)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.title(f'Top {top_n} Correlations with {churn_column.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add vertical line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add correlation values as text on bars
    for i, v in enumerate(top_correlations.values):
        plt.text(v + (0.01 if v > 0 else -0.01), i, f'{v:.3f}', 
                va='center', ha='left' if v > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print correlations
    print(f"\nTop {top_n} Correlations with {churn_column}:")
    print("-" * 50)
    for i, (feature, corr) in enumerate(top_correlations.items(), 1):
        direction = "↑ Increases churn" if corr > 0 else "↓ Decreases churn"
        print(f"{i:2d}. {feature:<25} {corr:6.3f} {direction}")
    
    return top_correlations

def prepare_correlation_data(df):
    """
    Prepare data by encoding categorical variables and creating useful features
    """
    analysis_df = df.copy()
    
    # Encode categorical variables
    categorical_columns = ['plan_type', 'contract_type', 'plan_tier']
    le = LabelEncoder()
    
    for col in categorical_columns:
        if col in analysis_df.columns:
            analysis_df[f'{col}_encoded'] = le.fit_transform(analysis_df[col].astype(str))
    
    # Create binary features from categorical data
    if 'plan_type' in analysis_df.columns:
        analysis_df['is_unlimited'] = analysis_df['plan_type'].str.contains('Unlimited', na=False).astype(int)
        analysis_df['is_premium'] = analysis_df['plan_type'].str.contains('Premium', na=False).astype(int)
        analysis_df['is_limited'] = analysis_df['plan_type'].str.contains('Limited', na=False).astype(int)
    
    if 'plan_tier' in analysis_df.columns:
        analysis_df['is_basic_plan'] = (analysis_df['plan_tier'] == 'Basic_25').astype(int)
        analysis_df['is_premium_plan'] = (analysis_df['plan_tier'] == 'Premium_Gig').astype(int)
    
    if 'contract_type' in analysis_df.columns:
        analysis_df['is_month_to_month'] = (analysis_df['contract_type'] == 'Month_to_Month').astype(int)
        analysis_df['is_long_contract'] = analysis_df['contract_type'].isin(['24_Month', '36_Month']).astype(int)
    
    # Create derived features for mobile
    if 'monthly_cost' in analysis_df.columns and 'line_count' in analysis_df.columns:
        analysis_df['cost_per_line'] = analysis_df['monthly_cost'] / analysis_df['line_count']
    
    if 'data_overage_frequency' in analysis_df.columns:
        analysis_df['high_overage'] = (analysis_df['data_overage_frequency'] > 2).astype(int)
        analysis_df['any_overage'] = (analysis_df['data_overage_frequency'] > 0).astype(int)
    
    # Create derived features for internet
    if 'data_usage_gb' in analysis_df.columns and 'speed_mbps' in analysis_df.columns:
        analysis_df['usage_per_mbps'] = analysis_df['data_usage_gb'] / analysis_df['speed_mbps']
    
    if 'speed_complaints' in analysis_df.columns:
        analysis_df['high_complaints'] = (analysis_df['speed_complaints'] > 1).astype(int)
    
    if 'outage_count' in analysis_df.columns:
        analysis_df['frequent_outages'] = (analysis_df['outage_count'] > 1).astype(int)
    
    # Create tenure-based features
    for tenure_col in ['mobile_tenure_days', 'internet_tenure_days']:
        if tenure_col in analysis_df.columns:
            analysis_df[f'long_tenure_{tenure_col.split("_")[0]}'] = (analysis_df[tenure_col] > 365).astype(int)
    
    # Handle boolean columns
    bool_columns = analysis_df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        analysis_df[col] = analysis_df[col].astype(int)
    
    return analysis_df



def display_cases(df_display, i=None):
    """
    Display customer call case information including transcript and follow-up email.
    
    Parameters:
    df_display (DataFrame): DataFrame containing customer call data
    i (int, optional): Index of the case to display. If None, selects a random case.
    """
    
    if i is None:
        # Fix: np.random.randint instead of np.rint for random integer selection
        i = np.random.randint(0, df_display.shape[0])
    
    # Ensure index is within bounds
    if i < 0 or i >= df_display.shape[0]:
        print(f"Error: Index {i} is out of bounds. DataFrame has {df_display.shape[0]} rows.")
        return
    
    print(f"Customer: {df_display.iloc[i]['customer_id']} called on {df_display.iloc[i]['call_date']}:")
    print("=" * 93)
    print("Hyperpersonalized Follow Up Email Generated:")
    print("=" * 93)
    pprint(df_display.iloc[i]['personalized_email'])
    print("=" * 93)
    print("Similar problems and Business solution for it:")
    print("=" * 93)
    pprint(df_display.iloc[i]['similar_matched_problem'])
    print("=" * 93)
    pprint(df_display.iloc[i]['recommended_solution'])
    print("=" * 93)
    print("Call Transcript:")
    print("=" * 93)
    pprint(df_display.iloc[i]['current_call_transcript'])
    print("=" * 93)
    
def display_agent_performance(df_agents, i=None):
    """
    Display agent performance evaluation including ratings, feedback, and coaching recommendations.
    
    Parameters:
    df_agents (DataFrame): DataFrame containing agent performance data
    i (int, optional): Index of the agent to display. If None, selects a random agent.
    """
    
    if i is None:
        # Select a random agent
        i = np.random.randint(0, df_agents.shape[0])
    
    # Ensure index is within bounds
    if i < 0 or i >= df_agents.shape[0]:
        print(f"Error: Index {i} is out of bounds. DataFrame has {df_agents.shape[0]} rows.")
        return
    
    # Get agent data
    agent_data = df_agents.iloc[i]
    
    print(f"Agent: {agent_data['agent_name']} (ID: {agent_data['agent_id']})")
    print(f"Evaluation Period: {agent_data['eval_week_start']} to {agent_data['eval_week_end']}")
    print(f"Average Rating: {agent_data['avg_rating']:.2f}")
    print(f"Successful Calls: {agent_data['successful_calls']} | Success Rate: {agent_data['success_rate_pct']:.1f}%")
    print("=" * 80)
    
    print("THINGS YOU EXCEL AT:")
    print("-" * 20)
    print(agent_data['weekly_positives'])
    print()
    
    print("THINGS YOU COULD IMPROVE ON:")
    print("-" * 30)
    print(agent_data['weekly_negatives'])
    print()
    
    print("HERE ARE SHORT RECOMMENDATIONS FOR IMPROVING AND GETTING THE BEST:")
    print("-" * 70)
    print(agent_data['coaching_recommendations'])
    print("=" * 80)
    
    
def display_customer_retention(df_customers, i=None):
    """
    Display customer retention analysis including problem matching, solutions, and personalized retention emails.
    
    Parameters:
    df_customers (DataFrame): DataFrame containing customer retention data
    i (int, optional): Index of the customer to display. If None, selects a random customer.
    """
    
    # Display total number of customers in the dataset
    print(f"Total Customers in Dataset: {df_customers.shape[0]}")
    print("=" * 80)
    print()
    
    if i is None:
        # Select a random customer
        i = np.random.randint(0, df_customers.shape[0])
    
    # Ensure index is within bounds
    if i < 0 or i >= df_customers.shape[0]:
        print(f"Error: Index {i} is out of bounds. DataFrame has {df_customers.shape[0]} rows.")
        return
    
    # Get customer data
    customer_data = df_customers.iloc[i]
    
    print(f"Customer: {customer_data['customer_id']}")
    print(f"Churn Risk: {customer_data['risk_category']} ({customer_data['churn_probability_percent']}%)")
    print(f"Internet Plan: {customer_data['internet_plan']} | Speed: {customer_data['internet_speed']}")
    print(f"Devices: {customer_data['device_types_concat']} | Security Devices: {customer_data['total_security_devices']}")
    print(f"Solution Match Score: {customer_data['solution_match_score']:.2f}")
    print("=" * 80)
    
    print("PROBLEM ANALYSIS:")
    print("-" * 20)
    pprint(customer_data['problem_analysis'])
    print()
    
    print("PERSONALIZED RETENTION EMAIL:")
    print("-" * 35)
    pprint(customer_data['personalized_retention_email'])
    print()
    
    print("TOP FEATURE ATTRIBUTIONS:")
    print("-" * 30)
    pprint(customer_data['top_feature_attributions'])
    print("=" * 80)