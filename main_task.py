import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Load the cleaned DataFrames

st.set_page_config(layout="wide")

st.title("Game User Behavior and Revenue Analysis Dashboard")

st.write("""
This dashboard provides insights into game user behavior, revenue trends, and characteristics of high-value and high-retention users.
""")

# Display key metrics (DAU, WAU, MAU)
st.header("Key Activity Metrics")

# Get the latest DAU, WAU, and MAU values

# Load the data (assuming the CSV is in the same directory)
try:
    # Update the file path to read from the 'game_dashboard' folder
    df = pd.read_csv("./game_dashboard/Matiks - Data Analyst Data - Sheet1.csv")

    # --- Data Cleaning and Preprocessing ---
    df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], format='%d-%b-%Y')
    df['Last_Login'] = pd.to_datetime(df['Last_Login'], format='%d-%b-%Y')
    df['Avg_Session_Duration_Min'] = df['Avg_Session_Duration_Min'].apply(lambda x: max(x, 0))
    df['Total_Hours_Played'] = df['Total_Hours_Played'].apply(lambda x: max(x, 0))

    # --- Feature Engineering ---
    df['Active_Days'] = (df['Last_Login'] - df['Signup_Date']).dt.days
    df['Total_Session_Duration_Hours'] = (df['Total_Play_Sessions'] * df['Avg_Session_Duration_Min']) / 60
    df['Revenue_per_Hour'] = df['Total_Revenue_USD'] / df['Total_Hours_Played']
    df['Revenue_per_Hour'] = df['Revenue_per_Hour'].replace([float('inf'), float('-inf')], 0)
    quantiles = df['Total_Revenue_USD'].quantile([0.33, 0.66]).to_list()

    def segment_user(revenue):
        if revenue <= quantiles[0]:
            return 'Low_Value'
        elif revenue <= quantiles[1]:
            return 'Medium_Value'
        else:
            return 'High_Value'
    df['User_Segment'] = df['Total_Revenue_USD'].apply(segment_user)

    # --- Calculate DAU, WAU, MAU ---
    dau = df.groupby(df['Last_Login'].dt.date)['User_ID'].nunique().reset_index()
    dau.columns = ['Date', 'DAU']
    df_wau_mau = df.set_index('Last_Login').sort_index()
    wau = df_wau_mau.resample('W')['User_ID'].apply(lambda x: x.unique().shape[0]).reset_index()
    wau.columns = ['Week', 'WAU']
    mau = df_wau_mau.resample('ME')['User_ID'].apply(lambda x: x.unique().shape[0]).reset_index()
    mau.columns = ['Month', 'MAU']

    # --- Analyze revenue trends ---
    daily_revenue = df.groupby(df['Last_Login'].dt.date)['Total_Revenue_USD'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Total_Revenue_USD']
    weekly_revenue = df.groupby(df['Last_Login'].dt.to_period('W'))['Total_Revenue_USD'].sum().reset_index()
    weekly_revenue['Last_Login'] = weekly_revenue['Last_Login'].astype(str)
    monthly_revenue = df.groupby(df['Last_Login'].dt.to_period('M'))['Total_Revenue_USD'].sum().reset_index()
    monthly_revenue['Last_Login'] = monthly_revenue['Last_Login'].astype(str)

    # --- Breakdown analysis ---
    device_metrics = df.groupby('Device_Type')[['Total_Revenue_USD', 'Total_Play_Sessions', 'Total_Session_Duration_Hours']].mean().reset_index()
    segment_metrics = df.groupby('User_Segment')[['Total_Revenue_USD', 'Total_Play_Sessions', 'Total_Session_Duration_Hours']].mean().reset_index()
    game_metrics = df.groupby('Game_Title')[['Total_Revenue_USD', 'Total_Play_Sessions', 'Total_Session_Duration_Hours']].mean().reset_index()

    # --- Identify behavioral patterns ---
    # Histograms are generated directly from df in the Streamlit app

    # --- Identify early signs of churn ---
    # Time difference analysis was not feasible with this data structure
    user_avg_session_duration = df.groupby('User_ID')['Avg_Session_Duration_Min'].mean().reset_index()
    user_avg_session_duration.columns = ['User_ID', 'Average_Session_Duration_Min']
    # Merge average session duration back to the main dataframe for scatter plot
    df_merged = pd.merge(df, user_avg_session_duration, on='User_ID', how='left')


    # --- Characterize high-value/high-retention users ---
    high_value_users = df[df['User_Segment'] == 'High_Value'].copy()
    retention_threshold = df['Active_Days'].quantile(0.75)
    high_retention_users = df[df['Active_Days'] >= retention_threshold].copy()
    numerical_features = ['Total_Play_Sessions', 'Total_Session_Duration_Hours', 'Total_Revenue_USD', 'Achievement_Score'] # Used for describe tables

    # --- Funnel tracking ---
    total_users = len(df)
    df['Reached_First_Session'] = df['Total_Play_Sessions'] > 0
    df['Made_First_Purchase'] = df['In_Game_Purchases_Count'] > 0
    users_reached_first_session = df['Reached_First_Session'].sum()
    users_made_first_purchase = df['Made_First_Purchase'].sum()
    funnel_steps = ['Signup', 'First Game Session', 'Made First Purchase']
    user_counts = [total_users, users_reached_first_session, users_made_first_purchase]


except FileNotFoundError:
    st.error("Error: Data file 'game_dashboard/Matiks - Data Analyst Data - Sheet1.csv' not found. Please make sure it's in the 'game_dashboard' directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during data processing: {e}")
    st.stop()


latest_dau = dau['DAU'].iloc[-1] if not dau.empty else 0
latest_wau = wau['WAU'].iloc[-1] if not wau.empty else 0
latest_mau = mau['MAU'].iloc[-1] if not mau.empty else 0


col1, col2, col3 = st.columns(3)
col1.metric("Latest Daily Active Users (DAU)", latest_dau)
col2.metric("Latest Weekly Active Users (WAU)", latest_wau)
col3.metric("Latest Monthly Active Users (MAU)", latest_mau)

# Create interactive visualizations for revenue trends
st.header("Revenue Trends Over Time")

revenue_granularity = st.selectbox(
    "Select Revenue Granularity",
    ('Daily', 'Weekly', 'Monthly')
)

# Adjusted figure size for matplotlib plots
plt.figure(figsize=(10, 5))
if revenue_granularity == 'Daily':
    st.subheader("Daily Revenue")
    plt.plot(daily_revenue['Date'], daily_revenue['Total_Revenue_USD'])
    plt.xlabel('Date')
    plt.ylabel('Total Revenue (USD)')
    plt.title('Daily Revenue Over Time')
elif revenue_granularity == 'Weekly':
    st.subheader("Weekly Revenue")
    plt.plot(weekly_revenue['Last_Login'], weekly_revenue['Total_Revenue_USD'])
    plt.xlabel('Week')
    plt.ylabel('Total Revenue (USD)')
    plt.title('Weekly Revenue Over Time')
else:
    st.subheader("Monthly Revenue")
    plt.plot(monthly_revenue['Last_Login'], monthly_revenue['Total_Revenue_USD'])
    plt.xlabel('Month')
    plt.ylabel('Total Revenue (USD)')
    plt.title('Monthly Revenue Over Time')

plt.xticks(rotation=45)
st.pyplot(plt)

# Create interactive breakdowns of key metrics
st.header("Metrics Breakdown")

breakdown_dimension = st.selectbox(
    "Select Breakdown Dimension",
    ('Device_Type', 'User_Segment', 'Game_Title')
)

st.subheader(f"Breakdown by {breakdown_dimension}")

# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
if breakdown_dimension == 'Device_Type':
    sns.barplot(data=device_metrics, x='Device_Type', y='Total_Revenue_USD')
    plt.title('Average Revenue by Device Type')
    plt.xlabel('Device Type')
    plt.ylabel('Average Total Revenue (USD)')
elif breakdown_dimension == 'User_Segment':
    sns.barplot(data=segment_metrics, x='User_Segment', y='Total_Revenue_USD', order=['Low_Value', 'Medium_Value', 'High_Value'])
    plt.title('Average Revenue by User Segment')
    plt.xlabel('User Segment')
    plt.ylabel('Average Total Revenue (USD)')
else: # Game_Title
    sns.barplot(data=game_metrics, x='Game_Title', y='Total_Revenue_USD')
    plt.title('Average Revenue by Game Title')
    plt.xlabel('Game Title')
    plt.ylabel('Average Total Revenue (USD)')

st.pyplot(plt)


# more breakdowns for other metrics (sessions, duration) similarly

# Include visualizations for behavioral patterns
st.header("Behavioral Patterns")

st.subheader("Distribution of Active Days")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.histplot(df['Active_Days'], bins=50, kde=True)
plt.title('Distribution of Active Days')
plt.xlabel('Active Days')
plt.ylabel('Number of Users')
st.pyplot(plt)

st.subheader("Distribution of Total Play Sessions")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.histplot(df['Total_Play_Sessions'], bins=50, kde=True)
plt.title('Distribution of Total Play Sessions')
plt.xlabel('Total Play Sessions')
plt.ylabel('Number of Users')
st.pyplot(plt)

st.subheader("Distribution of Total Session Duration (Hours)")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.histplot(df['Total_Session_Duration_Hours'], bins=50, kde=True)
plt.title('Distribution of Total Session Duration (Hours)')
plt.xlabel('Total Session Duration (Hours)')
plt.ylabel('Number of Users')
st.pyplot(plt)


# Include visualizations exploring the relationship between behavioral metrics and revenue/user segment
st.header("Behavioral Metrics vs. Revenue/User Segment")

st.subheader("Active Days vs. Total Revenue")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Active_Days', y='Total_Revenue_USD', alpha=0.6)
plt.title('Active Days vs. Total Revenue')
plt.xlabel('Active Days')
plt.ylabel('Total Revenue (USD)')
st.pyplot(plt)

st.subheader("Total Play Sessions vs. Total Revenue")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Total_Play_Sessions', y='Total_Revenue_USD', alpha=0.6)
plt.title('Total Play Sessions vs. Total Revenue')
plt.xlabel('Total Play Sessions')
plt.ylabel('Total Revenue (USD)')
st.pyplot(plt)

st.subheader("Total Session Duration (Hours) vs. Total Revenue")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Total_Session_Duration_Hours', y='Total_Revenue_USD', alpha=0.6)
plt.title('Total Session Duration (Hours) vs. Total Revenue')
plt.xlabel('Total Session Duration (Hours)')
plt.ylabel('Total Revenue (USD)')
st.pyplot(plt)

st.subheader("Total Revenue by User Segment")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='User_Segment', y='Total_Revenue_USD', order=['Low_Value', 'Medium_Value', 'High_Value'])
plt.title('Total Revenue by User Segment')
plt.xlabel('User Segment')
plt.ylabel('Total Revenue (USD)')
st.pyplot(plt)


# Include visualizations related to churn indicators
st.header("Churn Indicators")

# Assuming user_avg_session_duration dataframe is available
st.subheader("Distribution of Average Session Duration per User")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.histplot(user_avg_session_duration['Average_Session_Duration_Min'], bins=50, kde=True)
plt.title('Distribution of Average Session Duration per User')
plt.xlabel('Average Session Duration (Minutes)')
plt.ylabel('Number of Users')
st.pyplot(plt)

# Assuming df_merged is available with Average_Session_Duration_Min
st.subheader("Average Session Duration vs. Total Revenue")
# Adjusted figure size for matplotlib plots
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_merged, x='Average_Session_Duration_Min', y='Total_Revenue_USD', alpha=0.6)
plt.title('Average Session Duration vs. Total Revenue')
plt.xlabel('Average Session Duration (Minutes)')
plt.ylabel('Total Revenue (USD)')
st.pyplot(plt)


# Include sections/visualizations for high-value and high-retention users
st.header("High-Value and High-Retention Users")

st.subheader("Characteristics of High-Value Users")
st.write("Descriptive statistics for High-Value Users:")
st.dataframe(high_value_users[numerical_features].describe().T)


st.subheader("Characteristics of High-Retention Users")
st.write("Descriptive statistics for High-Retention Users:")
st.dataframe(high_retention_users[numerical_features].describe().T)

# Include Funnel Visualization
st.header("User Funnel Analysis")

# Calculate conversion rates for display
conversion_rates = [
    f"{(user_counts[i+1] / user_counts[i])*100:.2f}%" if user_counts[i] > 0 else "N/A"
    for i in range(len(user_counts) - 1)
]

fig = go.Figure(go.Funnel(
    y=funnel_steps,
    x=user_counts,
    textposition="inside",
    textinfo="value+percent initial",
    opacity=0.65,
    marker={"color": ["deepskyblue", "lightsalmon", "cadetblue"],
            "line": {"width": 3, "color": "darkgrey"}},
    connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
))

annotations = []
for i in range(len(funnel_steps) - 1):
    annotations.append(dict(
        x=user_counts[i+1],
        y=funnel_steps[i],
        text=f"Conversion: {conversion_rates[i]}",
        showarrow=False,
        xanchor='left',
        yanchor='bottom',
        font=dict(size=10, color="black")
    ))

# Adjusted layout for plotly figure size
fig.update_layout(
    title="User Funnel and Conversion Rates",
    annotations=annotations,
    width=700,  # Adjust width as needed
    height=500  # Adjust height as needed
)

st.plotly_chart(fig, use_container_width=True)
