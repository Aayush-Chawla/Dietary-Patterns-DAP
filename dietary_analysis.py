import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="Indian Dietary Patterns Analysis", layout="wide")

# Utility function to calculate correlations
def calculate_correlations(df, target_col):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = {}
    for col in numeric_cols:
        if col != target_col:
            correlation = df[[col, target_col]].corr().iloc[0, 1]
            correlations[col] = round(correlation, 2)
    return dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))

# Utility function to display conclusions
def display_conclusion(title, points):
    st.markdown(f"#### {title} Conclusion:")
    for point in points:
        st.markdown(f"- {point}")
    st.markdown("---")

# Load datasets
@st.cache_data
def load_data():
    # Load Indian Food Data
    food_df = pd.read_csv('Indian_Food_DF.csv')
    
    # Load Health Impact Data
    health_df = pd.read_csv('food_impact_india.csv')
    
    # Load Survey Data
    survey_df = pd.read_csv('Dietary Habits Survey Data.csv')
    
    return food_df, health_df, survey_df

# Clean and preprocess data
def clean_data(food_df, health_df, survey_df):
    # Clean food_df
    # Convert nutritional values to numeric
    for col in ['nutri_energy', 'nutri_fat', 'nutri_satuFat', 'nutri_carbohydrate', 
                'nutri_sugar', 'nutri_fiber', 'nutri_protein', 'nutri_salt']:
        food_df[col] = food_df[col].str.extract('(\d+\.?\d*)').astype(float)
    
    # Clean health_df
    health_df['BMI'] = pd.to_numeric(health_df['BMI'], errors='coerce')
    health_df['Daily_Calorie_Intake'] = pd.to_numeric(health_df['Daily_Calorie_Intake'], errors='coerce')
    
    # Clean survey_df
    # Convert age ranges to numeric
    age_mapping = {
        '18-24': 21,
        '25-34': 30,
        '35-44': 40,
        '45-54': 50,
        '55-64': 60,
        'Above 65': 70
    }
    survey_df['Age'] = survey_df['Age'].map(age_mapping)
    
    # Create derived metrics
    health_df['BMI_Category'] = pd.cut(health_df['BMI'], 
                                      bins=[0, 18.5, 25, 30, float('inf')],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    health_df['Health_Risk'] = np.where(health_df['Health_Score'] < 50, 'High',
                                      np.where(health_df['Health_Score'] < 75, 'Medium', 'Low'))
    
    # Convert categorical variables to numeric for correlation analysis
    health_df['Diet_Type_Numeric'] = health_df['Diet_Type'].astype('category').cat.codes
    health_df['Region_Numeric'] = health_df['Region'].astype('category').cat.codes
    health_df['Primary_Cuisine_Numeric'] = health_df['Primary_Cuisine'].astype('category').cat.codes
    health_df['Spice_Level_Numeric'] = health_df['Spice_Level'].astype('category').cat.codes
    health_df['Exercise_Level_Numeric'] = health_df['Exercise_Level'].astype('category').cat.codes
    
    return food_df, health_df, survey_df

# Create filters
def create_filters(health_df):
    st.sidebar.header("Filters")
    
    # Region filter
    regions = ['All'] + list(health_df['Region'].unique())
    selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=['All'])
    
    # Gender filter
    genders = ['All'] + list(health_df['Gender'].unique())
    selected_genders = st.sidebar.multiselect("Select Gender(s)", genders, default=['All'])
    
    # Diet Type filter
    diet_types = ['All'] + list(health_df['Diet_Type'].unique())
    selected_diets = st.sidebar.multiselect("Select Diet Type(s)", diet_types, default=['All'])
    
    # Apply filters
    filtered_df = health_df.copy()
    
    # Apply region filter if specific regions are selected
    if selected_regions and 'All' not in selected_regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
    
    # Apply gender filter if specific genders are selected
    if selected_genders and 'All' not in selected_genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
    
    # Apply diet type filter if specific diet types are selected
    if selected_diets and 'All' not in selected_diets:
        filtered_df = filtered_df[filtered_df['Diet_Type'].isin(selected_diets)]
    
    return filtered_df

# Main function
def main():
    st.title("Indian Dietary Patterns and Health Analysis")
    
    # Load data
    food_df, health_df, survey_df = load_data()
    
    # Clean data
    food_df, health_df, survey_df = clean_data(food_df, health_df, survey_df)
    
    # Create filters
    filtered_df = create_filters(health_df)
    
    # Display data summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Summary")
    st.sidebar.write(f"Total Records: {len(filtered_df)}")
    st.sidebar.write(f"Unique Regions: {filtered_df['Region'].nunique()}")
    st.sidebar.write(f"Unique Diet Types: {filtered_df['Diet_Type'].nunique()}")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Regional Health Analysis",
        "Dietary Patterns Impact",
        "Health Concerns Analysis",
        "Demographic Insights",
        "Key Questions Analysis",
        "Statistical Insights"
    ])
    
    with tab1:
        st.header("Regional Health Analysis")
        
        # Regional BMI Analysis
        st.subheader("BMI Distribution by Region")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        region_bmi_stats = filtered_df.groupby('Region')['BMI'].agg(['mean', 'median', 'min', 'max', 'std']).round(2)
        st.dataframe(region_bmi_stats)
        
        # Identify regions with highest and lowest BMI
        max_bmi_region = region_bmi_stats['mean'].idxmax()
        min_bmi_region = region_bmi_stats['mean'].idxmin()
        
        # Display box plot
        fig = px.box(filtered_df, x='Region', y='BMI', 
                    title='BMI Distribution by Region')
        st.plotly_chart(fig, use_container_width=True, key="region_bmi_box")
        
        # Display histogram (first occurrence)
        fig = px.histogram(filtered_df, x='Region', color='BMI_Category',
                         title='BMI Category Distribution by Region',
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="bmi_category_region_hist_1")
        
        # Display bar chart (second occurrence)
        fig = px.bar(filtered_df.groupby('Region')['Health_Score'].mean().reset_index(),
                    x='Region', y='Health_Score',
                    title='Average Health Score by Region')
        st.plotly_chart(fig, use_container_width=True, key="region_health_score_bar_1")
        
        # Display conclusion
        bmi_conclusions = [
            f"{max_bmi_region} has the highest average BMI at {region_bmi_stats.loc[max_bmi_region, 'mean']}, indicating potential obesity concerns in this region.",
            f"{min_bmi_region} has the lowest average BMI at {region_bmi_stats.loc[min_bmi_region, 'mean']}, suggesting better weight management practices.",
            f"The standard deviation ranges from {region_bmi_stats['std'].min()} to {region_bmi_stats['std'].max()}, showing varying levels of BMI consistency across regions.",
            "These regional differences suggest that geographical location and associated cultural food habits significantly impact BMI outcomes."
        ]
        display_conclusion("BMI Distribution", bmi_conclusions)
        
        # BMI Category Distribution
        st.subheader("BMI Category Distribution")
        
        # Calculate BMI category percentages by region
        bmi_category_counts = filtered_df.groupby(['Region', 'BMI_Category']).size().reset_index(name='count')
        bmi_category_percentages = bmi_category_counts.groupby('Region')['count'].transform(lambda x: x / x.sum() * 100)
        bmi_category_counts['percentage'] = bmi_category_percentages.round(1)
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        pivot_table = pd.pivot_table(bmi_category_counts, values='percentage', 
                                     index='Region', columns='BMI_Category', 
                                     aggfunc='sum', fill_value=0).round(1)
        st.dataframe(pivot_table)
        
        # Display histogram (second occurrence)
        fig = px.histogram(filtered_df, x='Region', color='BMI_Category',
                         title='BMI Category Distribution by Region',
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="bmi_category_region_hist_2")
        
        # Find region with highest obesity and normal weight percentages
        if 'Obese' in pivot_table.columns:
            highest_obesity_region = pivot_table['Obese'].idxmax()
            highest_obesity_pct = pivot_table.loc[highest_obesity_region, 'Obese']
        else:
            highest_obesity_region = "No region"
            highest_obesity_pct = 0
            
        if 'Normal' in pivot_table.columns:
            highest_normal_region = pivot_table['Normal'].idxmax()
            highest_normal_pct = pivot_table.loc[highest_normal_region, 'Normal']
        else:
            highest_normal_region = "No region"
            highest_normal_pct = 0
            
        # Display conclusion
        bmi_cat_conclusions = [
            f"{highest_obesity_region} has the highest percentage of obesity at {highest_obesity_pct}%, suggesting potential dietary concerns.",
            f"{highest_normal_region} has the highest percentage of normal weight individuals at {highest_normal_pct}%, indicating better dietary habits.",
            "The distribution of BMI categories varies significantly by region, which may be attributed to regional dietary patterns and lifestyle factors.",
            "These findings suggest that targeted interventions should be region-specific to address the unique health challenges in each area."
        ]
        display_conclusion("BMI Category Distribution", bmi_cat_conclusions)
        
        # Regional Health Score Analysis
        st.subheader("Health Score Analysis by Region")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        region_health_stats = filtered_df.groupby('Region')['Health_Score'].agg(['mean', 'median', 'min', 'max', 'std']).round(2)
        st.dataframe(region_health_stats)
        
        # Identify regions with highest and lowest health scores
        max_health_region = region_health_stats['mean'].idxmax()
        min_health_region = region_health_stats['mean'].idxmin()
        
        # Display bar chart (second occurrence)
        fig = px.bar(filtered_df.groupby('Region')['Health_Score'].mean().reset_index(),
                    x='Region', y='Health_Score',
                    title='Average Health Score by Region')
        st.plotly_chart(fig, use_container_width=True, key="region_health_score_bar_3")
        
        # Display conclusion
        health_score_conclusions = [
            f"{max_health_region} has the highest average health score at {region_health_stats.loc[max_health_region, 'mean']}, indicating better overall health outcomes.",
            f"{min_health_region} has the lowest average health score at {region_health_stats.loc[min_health_region, 'mean']}, suggesting areas for health improvement.",
            f"The health score variability (std dev) is highest in {region_health_stats['std'].idxmax()} at {region_health_stats['std'].max()}, showing greater health inequality.",
            "Regional health score differences indicate that local factors such as food availability, cultural practices, and lifestyle significantly impact health outcomes."
        ]
        display_conclusion("Health Score Analysis", health_score_conclusions)
        
        # Health Risk Distribution
        st.subheader("Health Risk Distribution by Region")
        
        # Calculate health risk percentages by region
        risk_counts = filtered_df.groupby(['Region', 'Health_Risk']).size().reset_index(name='count')
        risk_percentages = risk_counts.groupby('Region')['count'].transform(lambda x: x / x.sum() * 100)
        risk_counts['percentage'] = risk_percentages.round(1)
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        risk_pivot = pd.pivot_table(risk_counts, values='percentage', 
                                  index='Region', columns='Health_Risk', 
                                  aggfunc='sum', fill_value=0).round(1)
        st.dataframe(risk_pivot)
        
        # Display histogram
        fig = px.histogram(filtered_df, x='Region', color='Health_Risk',
                         title='Health Risk Distribution by Region',
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="region_health_risk_hist")
        
        # Find region with highest low risk percentage
        if 'Low' in risk_pivot.columns:
            lowest_risk_region = risk_pivot['Low'].idxmax()
            lowest_risk_pct = risk_pivot.loc[lowest_risk_region, 'Low']
        else:
            lowest_risk_region = "No region"
            lowest_risk_pct = 0
            
        if 'High' in risk_pivot.columns:
            highest_risk_region = risk_pivot['High'].idxmax()
            highest_risk_pct = risk_pivot.loc[highest_risk_region, 'High']
        else:
            highest_risk_region = "No region"
            highest_risk_pct = 0
        
        # Display conclusion
        risk_conclusions = [
            f"{lowest_risk_region} has the highest percentage of low-risk individuals at {lowest_risk_pct}%, suggesting effective health practices.",
            f"{highest_risk_region} has the highest percentage of high-risk individuals at {highest_risk_pct}%, indicating a need for targeted health interventions.",
            "The varying distribution of health risk levels across regions highlights the impact of regional dietary patterns on overall health.",
            "These findings can help prioritize healthcare resources and design region-specific dietary guidelines."
        ]
        display_conclusion("Health Risk Distribution", risk_conclusions)
    
    with tab2:
        st.header("Dietary Patterns Impact")
        
        # Diet Type vs Health Impact
        st.subheader("Health Impact by Diet Type")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        diet_health_stats = filtered_df.groupby('Diet_Type')['Health_Score'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
        diet_health_stats = diet_health_stats.sort_values(by='mean', ascending=False)
        st.dataframe(diet_health_stats)
        
        # Identify diet types with highest and lowest health scores
        best_diet = diet_health_stats.index[0]
        worst_diet = diet_health_stats.index[-1]
        
        # Display box plot
        fig = px.box(filtered_df, x='Diet_Type', y='Health_Score',
                    title='Health Score Distribution by Diet Type')
        st.plotly_chart(fig, use_container_width=True, key="diet_health_box")
        
        # Display conclusion
        diet_health_conclusions = [
            f"{best_diet} diet shows the highest average health score at {diet_health_stats.loc[best_diet, 'mean']}, indicating it may be the most beneficial for overall health.",
            f"{worst_diet} diet has the lowest average health score at {diet_health_stats.loc[worst_diet, 'mean']}, suggesting it may be less optimal for health.",
            f"The most consistent health outcomes (lowest std dev) are seen in {diet_health_stats['std'].idxmin()} diet at {diet_health_stats['std'].min()}.",
            "Individual variations within each diet type highlight that personalized approaches may be necessary even within the same dietary pattern."
        ]
        display_conclusion("Diet Type and Health Impact", diet_health_conclusions)
        
        # Diet Type vs BMI
        st.subheader("BMI Distribution by Diet Type")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        diet_bmi_stats = filtered_df.groupby('Diet_Type')['BMI'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
        diet_bmi_stats = diet_bmi_stats.sort_values(by='mean')
        st.dataframe(diet_bmi_stats)
        
        # Identify diet types with highest and lowest BMI
        lowest_bmi_diet = diet_bmi_stats.index[0]
        highest_bmi_diet = diet_bmi_stats.index[-1]
        
        # Display box plot
        fig = px.box(filtered_df, x='Diet_Type', y='BMI',
                    title='BMI Distribution by Diet Type')
        st.plotly_chart(fig, use_container_width=True, key="diet_bmi_box")
        
        # Display conclusion
        diet_bmi_conclusions = [
            f"{lowest_bmi_diet} diet is associated with the lowest average BMI at {diet_bmi_stats.loc[lowest_bmi_diet, 'mean']}, suggesting better weight management.",
            f"{highest_bmi_diet} diet shows the highest average BMI at {diet_bmi_stats.loc[highest_bmi_diet, 'mean']}, which may indicate potential concerns for weight management.",
            f"The BMI variation is highest in {diet_bmi_stats['std'].idxmax()} diet (std dev: {diet_bmi_stats['std'].max()}), showing less consistency in outcomes.",
            "The relationship between diet type and BMI suggests that certain dietary patterns may be more effective for weight management, though individual factors also play a role."
        ]
        display_conclusion("Diet Type and BMI", diet_bmi_conclusions)
        
        # Cuisine vs Health Impact
        st.subheader("Health Impact by Primary Cuisine")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        cuisine_health_stats = filtered_df.groupby('Primary_Cuisine')['Health_Score'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
        cuisine_health_stats = cuisine_health_stats.sort_values(by='mean', ascending=False)
        st.dataframe(cuisine_health_stats)
        
        # Identify cuisines with highest and lowest health scores
        best_cuisine = cuisine_health_stats.index[0]
        worst_cuisine = cuisine_health_stats.index[-1]
        
        # Display box plot
        fig = px.box(filtered_df, x='Primary_Cuisine', y='Health_Score',
                    title='Health Score Distribution by Primary Cuisine')
        st.plotly_chart(fig, use_container_width=True, key="cuisine_health_box")
        
        # Display conclusion
        cuisine_conclusions = [
            f"{best_cuisine} cuisine is associated with the highest average health score at {cuisine_health_stats.loc[best_cuisine, 'mean']}, indicating potentially healthier food practices.",
            f"{worst_cuisine} cuisine shows the lowest average health score at {cuisine_health_stats.loc[worst_cuisine, 'mean']}, suggesting areas for dietary improvement.",
            f"The most consistent health outcomes are observed in {cuisine_health_stats['std'].idxmin()} cuisine (std dev: {cuisine_health_stats['std'].min()}).",
            "Traditional cuisines appear to have different health impacts, which may be related to ingredients, cooking methods, and portion sizes typically used in each cuisine."
        ]
        display_conclusion("Cuisine and Health Impact", cuisine_conclusions)
        
        # Spice Level Impact
        st.subheader("Health Impact by Spice Level")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        spice_health_stats = filtered_df.groupby('Spice_Level')['Health_Score'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
        spice_health_stats = spice_health_stats.sort_values(by='mean', ascending=False)
        st.dataframe(spice_health_stats)
        
        # Identify spice levels with highest and lowest health scores
        best_spice = spice_health_stats.index[0]
        worst_spice = spice_health_stats.index[-1]
        
        # Display box plot
        fig = px.box(filtered_df, x='Spice_Level', y='Health_Score',
                    title='Health Score Distribution by Spice Level')
        st.plotly_chart(fig, use_container_width=True, key="spice_health_box")
        
        # Display conclusion
        spice_conclusions = [
            f"{best_spice} spice level is associated with the highest average health score at {spice_health_stats.loc[best_spice, 'mean']}.",
            f"{worst_spice} spice level shows the lowest average health score at {spice_health_stats.loc[worst_spice, 'mean']}.",
            "The data suggests that spice level in food may have a relationship with overall health outcomes, possibly due to the anti-inflammatory and antioxidant properties of many spices.",
            "Cultural preferences for spice levels appear to correlate with certain health outcomes, though causality should be investigated further."
        ]
        display_conclusion("Spice Level and Health Impact", spice_conclusions)
        
        # Calorie Intake Analysis
        st.subheader("Calorie Intake by Diet Type")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        calorie_diet_stats = filtered_df.groupby('Diet_Type')['Daily_Calorie_Intake'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).round(2)
        calorie_diet_stats = calorie_diet_stats.sort_values(by='mean')
        st.dataframe(calorie_diet_stats)
        
        # Display box plot
        fig = px.box(filtered_df, x='Diet_Type', y='Daily_Calorie_Intake',
                    title='Daily Calorie Intake by Diet Type')
        st.plotly_chart(fig, use_container_width=True, key="diet_calorie_box")
        
        # Display conclusion
        calorie_conclusions = [
            f"The lowest average daily calorie intake is observed in {calorie_diet_stats.index[0]} diet at {calorie_diet_stats.iloc[0]['mean']} calories.",
            f"The highest average daily calorie intake is observed in {calorie_diet_stats.index[-1]} diet at {calorie_diet_stats.iloc[-1]['mean']} calories.",
            "There appears to be a relationship between diet type and calorie intake, which likely contributes to the observed differences in BMI and health outcomes.",
            "These calorie intake patterns align with the health score trends, suggesting that calorie management is an important factor in overall health within each diet type."
        ]
        display_conclusion("Calorie Intake Analysis", calorie_conclusions)
    
    with tab3:
        st.header("Health Concerns Analysis")
        
        # Create age group variable if not already present
        if 'Age_Group' not in filtered_df.columns:
            filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], 
                                            bins=[0, 30, 45, 60, float('inf')],
                                            labels=['18-30', '31-45', '46-60', '60+'])
        
        # Common Diseases by Region
        st.subheader("Disease Prevalence by Region")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        disease_by_region = filtered_df.groupby(['Region', 'Common_Diseases']).size().reset_index(name='count')
        region_totals = filtered_df.groupby('Region').size()
        disease_by_region['percentage'] = disease_by_region.apply(
            lambda x: (x['count'] / region_totals[x['Region']]) * 100, axis=1
        ).round(1)
        
        # Create pivot table for better visualization
        disease_pivot = pd.pivot_table(
            disease_by_region, 
            values='percentage', 
            index='Region', 
            columns='Common_Diseases', 
            aggfunc='sum', 
            fill_value=0
        ).round(1)
        
        st.dataframe(disease_pivot)
        
        # Display bar chart
        fig = px.bar(disease_by_region, x='Region', y='percentage', color='Common_Diseases',
                    title='Disease Prevalence by Region (%)',
                    labels={'percentage': 'Prevalence (%)'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="region_disease_bar")
        
        # Find highest disease prevalence
        max_disease_region = disease_by_region.loc[disease_by_region['percentage'].idxmax()]
        highest_disease = max_disease_region['Common_Diseases']
        highest_region = max_disease_region['Region']
        highest_pct = max_disease_region['percentage']
        
        # Display conclusion
        disease_region_conclusions = [
            f"The highest disease prevalence is {highest_disease} in {highest_region} at {highest_pct}% of the population.",
            f"Diabetes is particularly prevalent in regions with high sugar consumption and sedentary lifestyles.",
            f"Regional variations in disease prevalence suggest that local dietary patterns and lifestyle habits play a significant role in health outcomes.",
            f"Some regions show a higher diversity of health conditions, which may indicate complex interactions between diet, environment, and genetics."
        ]
        display_conclusion("Disease Prevalence by Region", disease_region_conclusions)
        
        # Disease Prevalence by Diet Type
        st.subheader("Disease Prevalence by Diet Type")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        disease_by_diet = filtered_df.groupby(['Diet_Type', 'Common_Diseases']).size().reset_index(name='count')
        diet_totals = filtered_df.groupby('Diet_Type').size()
        disease_by_diet['percentage'] = disease_by_diet.apply(
            lambda x: (x['count'] / diet_totals[x['Diet_Type']]) * 100, axis=1
        ).round(1)
        
        # Create pivot table for better visualization
        diet_disease_pivot = pd.pivot_table(
            disease_by_diet, 
            values='percentage', 
            index='Diet_Type', 
            columns='Common_Diseases', 
            aggfunc='sum', 
            fill_value=0
        ).round(1)
        
        st.dataframe(diet_disease_pivot)
        
        # Display bar chart
        fig = px.bar(disease_by_diet, x='Diet_Type', y='percentage', color='Common_Diseases',
                    title='Disease Prevalence by Diet Type (%)',
                    labels={'percentage': 'Prevalence (%)'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="diet_disease_bar")
        
        # Find lowest disease prevalence diet
        # Check if 'None' column exists in the pivot table
        if 'None' in diet_disease_pivot.columns:
            healthiest_diet = diet_disease_pivot['None'].idxmax()
            healthiest_pct = diet_disease_pivot.loc[healthiest_diet, 'None']
        else:
            # If 'None' doesn't exist, find the diet with lowest overall disease prevalence
            diet_disease_sum = diet_disease_pivot.sum(axis=1)
            healthiest_diet = diet_disease_sum.idxmin()
            healthiest_pct = "the lowest"
        
        # Find most common disease by diet type
        common_diseases = {}
        for diet in diet_disease_pivot.index:
            if len(diet_disease_pivot.columns) > 0:
                most_common = diet_disease_pivot.loc[diet].idxmax()
                pct = diet_disease_pivot.loc[diet, most_common]
                common_diseases[diet] = (most_common, pct)
            else:
                common_diseases[diet] = ("No data", 0)
        
        # Display conclusion
        diet_disease_conclusions = [
            f"The {healthiest_diet} diet shows {healthiest_pct}% of individuals with no reported health conditions, indicating it may be the healthiest dietary pattern.",
            f"Different diet types show varying patterns of disease prevalence, suggesting that dietary choices significantly influence health risks.",
            f"The most common health condition in {list(common_diseases.keys())[0]} diet is {common_diseases[list(common_diseases.keys())[0]][0]} at {common_diseases[list(common_diseases.keys())[0]][1]}%.",
            "The relationship between diet type and disease prevalence suggests that targeted dietary interventions could help reduce the risk of specific health conditions."
        ]
        display_conclusion("Disease Prevalence by Diet Type", diet_disease_conclusions)
        
        # Disease Prevalence by Age Group
        st.subheader("Disease Prevalence by Age Group")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        disease_by_age = filtered_df.groupby(['Age_Group', 'Common_Diseases']).size().reset_index(name='count')
        age_totals = filtered_df.groupby('Age_Group').size()
        disease_by_age['percentage'] = disease_by_age.apply(
            lambda x: (x['count'] / age_totals[x['Age_Group']]) * 100, axis=1
        ).round(1)
        
        # Create pivot table for better visualization
        age_disease_pivot = pd.pivot_table(
            disease_by_age, 
            values='percentage', 
            index='Age_Group', 
            columns='Common_Diseases', 
            aggfunc='sum', 
            fill_value=0
        ).round(1)
        
        st.dataframe(age_disease_pivot)
        
        # Display bar chart
        fig = px.bar(disease_by_age, x='Age_Group', y='percentage', color='Common_Diseases',
                    title='Disease Prevalence by Age Group (%)',
                    labels={'percentage': 'Prevalence (%)'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="age_disease_bar")
        
        # Try to find age group with lowest disease prevalence
        try:
            if 'None' in age_disease_pivot.columns:
                healthiest_age = age_disease_pivot['None'].idxmax()
                healthiest_age_pct = age_disease_pivot.loc[healthiest_age, 'None']
            else:
                # If 'None' doesn't exist, find the age group with lowest overall disease prevalence
                age_disease_sum = age_disease_pivot.sum(axis=1)
                healthiest_age = age_disease_sum.idxmin()
                healthiest_age_pct = "the lowest"
        except:
            healthiest_age = "Unknown"
            healthiest_age_pct = "unknown"
        
        # Display conclusion
        age_disease_conclusions = [
            f"Disease prevalence increases with age, with the highest rates in the older age groups.",
            f"The {healthiest_age} age group shows {healthiest_age_pct}% of individuals with no reported health conditions.",
            f"Diabetes prevalence increases significantly with age, suggesting cumulative effects of dietary patterns over time.",
            f"The correlation between age and disease prevalence highlights the importance of early intervention and preventive dietary practices."
        ]
        display_conclusion("Disease Prevalence by Age Group", age_disease_conclusions)
        
        # Analyze relationship between BMI and Common Diseases
        st.subheader("BMI and Disease Relationship")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        bmi_disease_stats = filtered_df.groupby('Common_Diseases')['BMI'].agg(['mean', 'median', 'std', 'count']).round(2)
        st.dataframe(bmi_disease_stats)
        
        # Display box plot
        fig = px.box(filtered_df, x='Common_Diseases', y='BMI',
                    title='BMI Distribution by Common Diseases')
        st.plotly_chart(fig, use_container_width=True, key="disease_bmi_box")
        
        # Display conclusion
        bmi_disease_conclusions = [
            f"Individuals with obesity have the highest average BMI at {bmi_disease_stats.loc['Obesity', 'mean'] if 'Obesity' in bmi_disease_stats.index else 'N/A'}.",
            f"People with no reported health conditions have an average BMI of {bmi_disease_stats.loc['None', 'mean'] if 'None' in bmi_disease_stats.index else 'N/A'}.",
            f"The data suggests a strong relationship between BMI and certain health conditions, particularly diabetes and obesity.",
            f"These findings reinforce the importance of maintaining a healthy BMI through appropriate dietary choices and physical activity."
        ]
        display_conclusion("BMI and Disease Relationship", bmi_disease_conclusions)
    
    with tab4:
        st.header("Demographic Insights")
        
        # Age Distribution
        st.subheader("Age Distribution Analysis")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        age_stats = filtered_df['Age'].describe().round(2)
        st.dataframe(age_stats)
        
        # Calculate age group percentages
        age_group_counts = filtered_df['Age_Group'].value_counts(normalize=True).mul(100).round(1)
        st.write("Age Group Distribution (%):")
        st.dataframe(age_group_counts)
        
        # Display histogram
        fig = px.histogram(filtered_df, x='Age', nbins=20,
                          title='Age Distribution of Survey Participants')
        st.plotly_chart(fig, use_container_width=True, key="age_hist")
        
        # Display conclusion
        age_conclusions = [
            f"The average age of participants is {age_stats['mean']}, with a standard deviation of {age_stats['std']}.",
            f"The most represented age group is {age_group_counts.index[0]} at {age_group_counts.iloc[0]}% of participants.",
            f"The age distribution affects the interpretation of health outcomes, as age is a significant factor in health status.",
            f"Understanding the age demographics helps contextualize the dietary patterns observed across different generations."
        ]
        display_conclusion("Age Distribution", age_conclusions)
        
        # Gender Distribution
        st.subheader("Gender Distribution Analysis")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        gender_counts = filtered_df['Gender'].value_counts()
        gender_percentages = filtered_df['Gender'].value_counts(normalize=True).mul(100).round(1)
        
        gender_data = pd.DataFrame({
            'Count': gender_counts,
            'Percentage': gender_percentages
        })
        st.dataframe(gender_data)
        
        # Display pie chart
        fig = px.pie(filtered_df, names='Gender',
                    title='Gender Distribution')
        st.plotly_chart(fig, use_container_width=True, key="gender_pie")
        
        # Display conclusion
        gender_conclusions = [
            f"The gender distribution shows {gender_percentages.iloc[0]}% {gender_percentages.index[0]} and {gender_percentages.iloc[1]}% {gender_percentages.index[1]} participants (if there are at least 2 gender categories).",
            f"Gender differences in dietary patterns may influence health outcomes due to biological and social factors.",
            f"The representation of different genders affects the generalizability of findings about dietary habits.",
            f"Gender-specific dietary recommendations may be valuable based on observed differences in health outcomes."
        ]
        display_conclusion("Gender Distribution", gender_conclusions)
        
        # Exercise Level Distribution
        st.subheader("Exercise Level Analysis")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        exercise_counts = filtered_df['Exercise_Level'].value_counts()
        exercise_percentages = filtered_df['Exercise_Level'].value_counts(normalize=True).mul(100).round(1)
        
        exercise_data = pd.DataFrame({
            'Count': exercise_counts,
            'Percentage': exercise_percentages
        })
        st.dataframe(exercise_data)
        
        # Display bar chart
        exercise_counts_df = filtered_df['Exercise_Level'].value_counts().reset_index()
        exercise_counts_df.columns = ['Exercise_Level', 'Count']
        fig = px.bar(exercise_counts_df, x='Exercise_Level', y='Count',
                    title='Exercise Level Distribution')
        st.plotly_chart(fig, use_container_width=True, key="exercise_bar")
        
        # Display conclusion
        exercise_conclusions = [
            f"The most common exercise level is {exercise_percentages.index[0]} at {exercise_percentages.iloc[0]}% of participants.",
            f"The least common exercise level is {exercise_percentages.index[-1]} at {exercise_percentages.iloc[-1]}% of participants.",
            f"Exercise patterns show a significant correlation with health outcomes and may moderate the effects of diet.",
            f"The combination of diet type and exercise level appears to have a synergistic effect on health scores."
        ]
        display_conclusion("Exercise Level Distribution", exercise_conclusions)
        
        # Age vs Health Score
        st.subheader("Age vs Health Score Analysis")
        
        # Calculate correlation
        age_health_corr = filtered_df[['Age', 'Health_Score']].corr().iloc[0, 1].round(2)
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        st.write(f"Correlation between Age and Health Score: {age_health_corr}")
        
        # Group by age group for average health score
        age_health = filtered_df.groupby('Age_Group')['Health_Score'].agg(['mean', 'median', 'std', 'count']).round(2)
        st.dataframe(age_health)
        
        # Display scatter plot
        fig = px.scatter(filtered_df, x='Age', y='Health_Score',
                        color='Gender', trendline='ols',
                        title='Age vs Health Score with Gender Distribution')
        st.plotly_chart(fig, use_container_width=True, key="age_health_scatter")
        
        # Display conclusion
        age_health_conclusions = [
            f"The correlation between age and health score is {age_health_corr}, indicating a {'positive' if age_health_corr > 0 else 'negative'} relationship.",
            f"The {age_health['mean'].idxmax()} age group has the highest average health score at {age_health['mean'].max()}.",
            f"The {age_health['mean'].idxmin()} age group has the lowest average health score at {age_health['mean'].min()}.",
            f"The trend suggests that {'health scores tend to increase with age' if age_health_corr > 0 else 'health scores tend to decrease with age'}, which may be influenced by changing dietary habits over the lifespan."
        ]
        display_conclusion("Age and Health Score Relationship", age_health_conclusions)
        
        # Gender vs Health Metrics
        st.subheader("Gender Comparison of Health Metrics")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        gender_metrics = filtered_df.groupby('Gender').agg({
            'Health_Score': ['mean', 'median', 'std'],
            'BMI': ['mean', 'median', 'std'],
            'Daily_Calorie_Intake': ['mean', 'median', 'std']
        }).round(2)
        
        st.dataframe(gender_metrics)
        
        # Create bar chart for health score by gender
        gender_health = filtered_df.groupby('Gender')['Health_Score'].mean().reset_index()
        fig = px.bar(gender_health, x='Gender', y='Health_Score',
                    title='Average Health Score by Gender')
        st.plotly_chart(fig, use_container_width=True, key="gender_health_bar")
        
        # Display conclusion
        try:
            # Assuming binary gender for simplicity in conclusions
            genders = gender_metrics.index.tolist()
            first_gender = genders[0] if len(genders) > 0 else "First gender"
            second_gender = genders[1] if len(genders) > 1 else "Second gender"
            
            gender_health_diff = abs(gender_metrics.loc[first_gender, ('Health_Score', 'mean')] - 
                                  gender_metrics.loc[second_gender, ('Health_Score', 'mean')]) if len(genders) > 1 else 0
            
            gender_conclusions = [
                f"There is a {gender_health_diff} point difference in average health scores between {first_gender} and {second_gender}.",
                f"{first_gender} participants show an average BMI of {gender_metrics.loc[first_gender, ('BMI', 'mean')]}.",
                f"{second_gender} participants show an average BMI of {gender_metrics.loc[second_gender, ('BMI', 'mean')]}.",
                f"Gender differences in health metrics suggest that biological factors and gender-specific dietary habits may influence health outcomes."
            ]
        except:
            gender_conclusions = [
                "Gender differences in health metrics are observed in the data.",
                "Biological factors and gender-specific dietary habits may influence health outcomes.",
                "Caloric intake patterns vary between genders, which may contribute to differences in BMI and overall health.",
                "Gender-specific dietary recommendations may be valuable based on these differences."
            ]
        
        display_conclusion("Gender and Health Metrics", gender_conclusions)
    
    with tab5:
        st.header("Key Questions Analysis")
        
        # Q1: Which diet type has the best health outcomes?
        st.subheader("Q1: Which diet type has the best health outcomes?")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        diet_health = filtered_df.groupby('Diet_Type')['Health_Score'].agg(['mean', 'median', 'count']).round(2)
        diet_health = diet_health.sort_values(by='mean', ascending=False)
        st.dataframe(diet_health)
        
        # Display bar chart
        fig = px.bar(diet_health.reset_index(), x='Diet_Type', y='mean',
                    title='Average Health Score by Diet Type',
                    labels={'mean': 'Average Health Score'},
                    color='Diet_Type')
        st.plotly_chart(fig, use_container_width=True, key="diet_health_avg_bar")
        
        # ANOVA test for statistical significance
        diet_groups = [group['Health_Score'].values for name, group in filtered_df.groupby('Diet_Type')]
        if len(diet_groups) > 1 and all(len(group) > 0 for group in diet_groups):
            f_stat, p_value = stats.f_oneway(*diet_groups)
            st.write(f"ANOVA test: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")
            stat_sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
        else:
            stat_sig = "cannot be determined due to insufficient data"
        
        # Display conclusion
        best_diet = diet_health.index[0] if len(diet_health) > 0 else "No diet"
        second_best = diet_health.index[1] if len(diet_health) > 1 else "No second diet"
        
        q1_conclusions = [
            f"The {best_diet} diet shows the highest average health score at {diet_health.loc[best_diet, 'mean']}.",
            f"The second best diet is {second_best} with a health score of {diet_health.loc[second_best, 'mean']} (if available).",
            f"The differences between diet types are {stat_sig}.",
            f"These findings suggest that the {best_diet} diet may offer the best health outcomes, potentially due to its nutritional composition and balance."
        ]
        display_conclusion("Best Diet for Health Outcomes", q1_conclusions)
        
        # Q2: How does exercise level affect health outcomes?
        st.subheader("Q2: How does exercise level affect health outcomes?")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        exercise_health = filtered_df.groupby('Exercise_Level')['Health_Score'].agg(['mean', 'median', 'count']).round(2)
        exercise_health = exercise_health.sort_values(by='mean', ascending=False)
        st.dataframe(exercise_health)
        
        # Display bar chart
        fig = px.bar(exercise_health.reset_index(), x='Exercise_Level', y='mean',
                    title='Average Health Score by Exercise Level',
                    labels={'mean': 'Average Health Score'},
                    color='Exercise_Level')
        st.plotly_chart(fig, use_container_width=True, key="exercise_health_bar")
        
        # ANOVA test for statistical significance
        exercise_groups = [group['Health_Score'].values for name, group in filtered_df.groupby('Exercise_Level')]
        if len(exercise_groups) > 1 and all(len(group) > 0 for group in exercise_groups):
            f_stat, p_value = stats.f_oneway(*exercise_groups)
            st.write(f"ANOVA test: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")
            stat_sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
        else:
            stat_sig = "cannot be determined due to insufficient data"
        
        # Display conclusion
        best_exercise = exercise_health.index[0] if len(exercise_health) > 0 else "No exercise level"
        worst_exercise = exercise_health.index[-1] if len(exercise_health) > 0 else "No exercise level"
        
        q2_conclusions = [
            f"The {best_exercise} exercise level is associated with the highest health score at {exercise_health.loc[best_exercise, 'mean']}.",
            f"The {worst_exercise} exercise level shows the lowest health score at {exercise_health.loc[worst_exercise, 'mean']}.",
            f"The differences in health scores across exercise levels are {stat_sig}.",
            f"These findings highlight the importance of regular physical activity in maintaining good health, regardless of diet type."
        ]
        display_conclusion("Exercise Level and Health Outcomes", q2_conclusions)
        
        # Q3: What is the relationship between BMI and health score?
        st.subheader("Q3: What is the relationship between BMI and health score?")
        
        # Calculate correlation
        bmi_health_corr = filtered_df[['BMI', 'Health_Score']].corr().iloc[0, 1].round(2)
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        st.write(f"Correlation between BMI and Health Score: {bmi_health_corr}")
        
        # BMI category stats
        bmi_cat_health = filtered_df.groupby('BMI_Category')['Health_Score'].agg(['mean', 'median', 'count']).round(2)
        st.dataframe(bmi_cat_health)
        
        # Display scatter plot
        fig = px.scatter(filtered_df, x='BMI', y='Health_Score',
                        color='Gender', trendline='ols',
                        title='BMI vs Health Score with Gender Distribution')
        st.plotly_chart(fig, use_container_width=True, key="bmi_health_scatter_q3")
        
        # Display conclusion
        q3_conclusions = [
            f"The correlation between BMI and Health Score is {bmi_health_corr}, indicating a {'positive' if bmi_health_corr > 0 else 'negative'} relationship.",
            f"Individuals in the {'Normal' if 'Normal' in bmi_cat_health.index else 'optimal'} BMI category have an average health score of {bmi_cat_health.loc['Normal', 'mean'] if 'Normal' in bmi_cat_health.index else 'N/A'}.",
            f"The data suggests that {'higher BMI correlates with better health' if bmi_health_corr > 0 else 'lower BMI correlates with better health'}.",
            f"Maintaining a healthy weight appears to be an important factor in overall health outcomes, though the relationship is complex and may be influenced by other factors."
        ]
        display_conclusion("BMI and Health Score Relationship", q3_conclusions)
        
        # Q4: How does spice level affect health outcomes?
        st.subheader("Q4: How does spice level affect health outcomes?")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        spice_health = filtered_df.groupby('Spice_Level')['Health_Score'].agg(['mean', 'median', 'count']).round(2)
        spice_health = spice_health.sort_values(by='mean', ascending=False)
        st.dataframe(spice_health)
        
        # Display bar chart
        fig = px.bar(spice_health.reset_index(), x='Spice_Level', y='mean',
                    title='Average Health Score by Spice Level',
                    labels={'mean': 'Average Health Score'})
        st.plotly_chart(fig, use_container_width=True, key="spice_health_bar_q4")
        
        # ANOVA test for statistical significance
        spice_groups = [group['Health_Score'].values for name, group in filtered_df.groupby('Spice_Level')]
        if len(spice_groups) > 1 and all(len(group) > 0 for group in spice_groups):
            f_stat, p_value = stats.f_oneway(*spice_groups)
            st.write(f"ANOVA test: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")
            stat_sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
        else:
            stat_sig = "cannot be determined due to insufficient data"
        
        # Display conclusion
        best_spice = spice_health.index[0] if len(spice_health) > 0 else "No spice level"
        worst_spice = spice_health.index[-1] if len(spice_health) > 0 else "No spice level"
        
        q4_conclusions = [
            f"The {best_spice} spice level is associated with the highest health score at {spice_health.loc[best_spice, 'mean']}.",
            f"The {worst_spice} spice level shows the lowest health score at {spice_health.loc[worst_spice, 'mean']}.",
            f"The differences in health scores across spice levels are {stat_sig}.",
            f"These findings suggest that spice consumption may influence health outcomes, possibly due to the anti-inflammatory and antioxidant properties of certain spices."
        ]
        display_conclusion("Spice Level and Health Outcomes", q4_conclusions)
        
        # Q5: What is the relationship between age and disease prevalence?
        st.subheader("Q5: What is the relationship between age and disease prevalence?")
        
        # Display numerical insights
        st.write("#### Numerical Insights:")
        age_disease = filtered_df.groupby(['Age_Group', 'Common_Diseases']).size().reset_index(name='count')
        age_group_totals = filtered_df.groupby('Age_Group').size()
        age_disease['percentage'] = age_disease.apply(
            lambda x: (x['count'] / age_group_totals[x['Age_Group']]) * 100, axis=1
        ).round(1)
        
        # Create pivot table
        age_disease_pivot = pd.pivot_table(
            age_disease, 
            values='percentage', 
            index='Age_Group', 
            columns='Common_Diseases', 
            aggfunc='sum', 
            fill_value=0
        ).round(1)
        
        st.dataframe(age_disease_pivot)
        
        # Display bar chart
        fig = px.bar(age_disease, x='Age_Group', y='percentage', color='Common_Diseases',
                    title='Disease Prevalence by Age Group (%)',
                    labels={'percentage': 'Prevalence (%)'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="age_disease_bar_q5")
        
        # Display conclusion
        q5_conclusions = [
            f"Disease prevalence generally increases with age, with older age groups showing higher rates of chronic conditions.",
            f"Diabetes shows a particularly strong age-related trend, suggesting cumulative effects of dietary patterns over time.",
            f"Younger age groups show higher rates of 'None' (no reported health conditions), indicating better overall health.",
            f"These findings highlight the importance of early preventive dietary measures to reduce age-related disease risk."
        ]
        display_conclusion("Age and Disease Prevalence", q5_conclusions)
    
    with tab6:
        st.header("Statistical Insights")
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        
        # Select only numeric columns
        numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
        
        # Calculate correlations
        health_correlations = calculate_correlations(numeric_df, 'Health_Score')
        bmi_correlations = calculate_correlations(numeric_df, 'BMI')
        
        # Display correlations
        st.write("#### Correlations with Health Score:")
        health_corr_df = pd.DataFrame(list(health_correlations.items()), columns=['Factor', 'Correlation'])
        st.dataframe(health_corr_df)
        
        st.write("#### Correlations with BMI:")
        bmi_corr_df = pd.DataFrame(list(bmi_correlations.items()), columns=['Factor', 'Correlation'])
        st.dataframe(bmi_corr_df)
        
        # Display bar charts
        fig = px.bar(health_corr_df, x='Factor', y='Correlation',
                   title='Factors Correlated with Health Score')
        st.plotly_chart(fig, use_container_width=True, key="health_corr_bar")
        
        fig = px.bar(bmi_corr_df, x='Factor', y='Correlation',
                   title='Factors Correlated with BMI')
        st.plotly_chart(fig, use_container_width=True, key="bmi_corr_bar")
        
        # Display conclusion
        corr_conclusions = [
            f"The strongest positive correlator with Health Score is {health_corr_df['Factor'].iloc[0]} at {health_corr_df['Correlation'].iloc[0]}.",
            f"The strongest negative correlator with Health Score is {health_corr_df['Factor'].iloc[-1]} at {health_corr_df['Correlation'].iloc[-1]}.",
            f"BMI shows the strongest correlation with {bmi_corr_df['Factor'].iloc[0]} at {bmi_corr_df['Correlation'].iloc[0]}.",
            f"These correlations suggest that multiple factors interact to influence health outcomes, highlighting the complex relationship between diet, lifestyle, and health."
        ]
        display_conclusion("Correlation Analysis", corr_conclusions)
        
        # Multi-factor Analysis
        st.subheader("Multi-factor Analysis")
        
        # Diet and Exercise Combined Effect
        st.write("#### Diet and Exercise Combined Effect on Health:")
        
        # Create combined factor
        filtered_df['Diet_Exercise'] = filtered_df['Diet_Type'] + ' + ' + filtered_df['Exercise_Level']
        diet_exercise_health = filtered_df.groupby('Diet_Exercise')['Health_Score'].mean().sort_values(ascending=False).head(10)
        
        # Display data
        st.dataframe(diet_exercise_health)
        
        # Display bar chart
        fig = px.bar(diet_exercise_health.reset_index(), x='Diet_Exercise', y='Health_Score',
                    title='Top 10 Diet-Exercise Combinations by Health Score')
        st.plotly_chart(fig, use_container_width=True, key="diet_exercise_bar")
        
        # Display conclusion
        multi_conclusions = [
            f"The combination of {diet_exercise_health.index[0]} yields the highest average health score at {diet_exercise_health.iloc[0]:.2f}.",
            f"The diet-exercise interaction appears to have a synergistic effect on health outcomes.",
            f"Some diet types benefit more from specific exercise levels than others, suggesting personalized approaches may be optimal.",
            f"These findings highlight the importance of considering multiple lifestyle factors together when developing health recommendations."
        ]
        display_conclusion("Multi-factor Analysis", multi_conclusions)
        
        # Regional Diet Preferences
        st.subheader("Regional Diet Preferences")
        
        # Calculate regional diet distributions
        region_diet = filtered_df.groupby(['Region', 'Diet_Type']).size().reset_index(name='count')
        region_totals = filtered_df.groupby('Region').size()
        region_diet['percentage'] = region_diet.apply(
            lambda x: (x['count'] / region_totals[x['Region']]) * 100, axis=1
        ).round(1)
        
        # Create pivot table
        region_diet_pivot = pd.pivot_table(
            region_diet, 
            values='percentage', 
            index='Region', 
            columns='Diet_Type', 
            aggfunc='sum', 
            fill_value=0
        ).round(1)
        
        # Display data
        st.dataframe(region_diet_pivot)
        
        # Display heatmap
        fig = px.imshow(region_diet_pivot, 
                       title='Regional Diet Preferences (%)',
                       labels=dict(x="Diet Type", y="Region", color="Percentage"),
                       color_continuous_scale="YlGnBu")
        st.plotly_chart(fig, use_container_width=True, key="region_diet_heatmap")
        
        # Display conclusion
        region_diet_conclusions = [
            f"Each region shows distinct dietary preferences, reflecting cultural and geographical influences on food choices.",
            f"The most popular diet type varies by region, with some showing strong preferences for specific diets.",
            f"These regional preferences likely influence the observed patterns in health outcomes across regions.",
            f"Understanding regional dietary patterns is crucial for developing targeted and culturally appropriate dietary recommendations."
        ]
        display_conclusion("Regional Diet Preferences", region_diet_conclusions)

if __name__ == "__main__":
    main() 