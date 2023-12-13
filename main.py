# -*- coding: utf-8 -*-
"""
Created on Tue Dec 07 16:23:42 2023

@author: Hafeez Ali
"""

# Importing required modules
import stats
import pandas as pd
import matplotlib.pyplot as plt


def file_to_dfs(filename):
    """
    This function will import the csv file into the program and read it in
    the World Bank format by skiping first rows and remove the columns
    not needed in the program. It'll clean the dataset and take transpose
    to return two dataframes.

    Parameters:
    - filename (str): The input DataFrame containing countries data.

    Returns:
    df1, df2 (pd.DataFrame): Dataframe containing the indicator data
                             and transposed version of the dataframe.
    """

    # Read the CSV file and skip the first four rows
    df1 = pd.read_csv(filename, skiprows=4)

    # Set the list of country names as the index of the dataframe
    df1 = df1.set_index(pd.Index(list(df1['Country Name'])))

    # Remove the last column that is not needed
    df1 = df1.iloc[:, :-1]

    # Clean the dataset from the columns not needed
    df1 = df1.drop(columns=['Country Name', 'Country Code',
                            'Indicator Code', 'Indicator Name'])

    # Taking transpose of the dataframe
    df2 = df1.T

    # Return the dataframes
    return df1, df2


# Calling the function to read the required csv files
df_forest_land_percent_1, df_forest_land_percent_2 = file_to_dfs(
    'API_AG.LND.FRST.ZS_DS2_en_csv_v2_6224694.csv')

df_power_consump_1, df_power_consump_2 = file_to_dfs(
    'API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_6229098.csv')

df_co2_emm_1, df_co2_emm_2 = file_to_dfs(
    'API_EN.ATM.CO2E.KT_DS2_en_csv_v2_6224818.csv')

df_urban_pop_1, df_urban_pop_2 = file_to_dfs(
    'API_SP.URB.TOTL_DS2_en_csv_v2_6227010.csv')

# Using describe method to get the statistical summary of the dataset
print('\nStatistical properties of CO2 emission level in Kt of World')
print(df_co2_emm_2['World'].describe())

# Importing and using the stats function skew to get skew value
print('\nCentralised and normalised skewness of CO2 emission level of World')
print(round(stats.skew(df_co2_emm_2['World']), 3))

# Importing and using the stats function kurtosis to get kurtosis value
print('\nCentralised and normalised kurtosis of CO2 emission level of World')
print(round(stats.kurtosis(df_co2_emm_2['World']), 3))

# Defining the years and countries list
years = ['1990', '1995', '2000', '2005', '2010', '2015', '2020']
countries = ['United Kingdom', 'France', 'United States',
             'China', 'India', 'Germany', 'Japan', 'Russian Federation']

# Select the values from the dataframe using .loc function
df_selected = df_co2_emm_2.loc[years, countries]

# Plotting the bar plot for dataframe
plt.figure(1)
ax = df_selected.plot(kind='bar', figsize=(10, 6), rot=45)
# Defining x and y label and title
ax.set_xlabel('Countries')
ax.set_ylabel('CO2 Emissions')
ax.set_title('Bar Plot of CO2 Emissions for different countries')
plt.legend(title='Carbon Emission in Kt')
plt.show()


df_selected = df_urban_pop_2.loc[years, countries]

# Plotting the bar plot for dataframe
plt.figure(2)
ax = df_selected.plot(kind='bar', figsize=(10, 6), rot=45)
# Defining x and y label and title
ax.set_xlabel('Countries')
ax.set_ylabel('Urban Population')
ax.set_title('Bar Plot of urban population for different countries')
plt.legend(title='Total Urban Population')
plt.show()


df_selected = df_power_consump_2.loc[years, countries]

# Plotting the bar plot for dataframe
plt.figure(2)
ax = df_selected.plot(kind='bar', figsize=(10, 6), rot=45)
# Defining x and y label and title
ax.set_xlabel('Countries')
ax.set_ylabel('Electricity Consumption in KWhr')
ax.set_title('Bar Plot of Electricity Consumption for different countries')
plt.legend(title='Electricity Consumption in KWhr')
plt.show()

# Plotting the scatter plot for dataframe
plt.figure(3)
plt.scatter(df_urban_pop_1.loc['World'], df_co2_emm_1.loc['World'])
# Defining x and y label and title
plt.xlabel('Urban Population')
plt.ylabel('CO2 Emissions (Kt)')
plt.title("World's Urban Population vs CO2 Emissions")
plt.show()

# Plotting the scatter plot for dataframe
plt.figure(4)
plt.scatter(df_power_consump_1.loc['World'], df_co2_emm_1.loc['World'])
# Defining x and y label and title
plt.xlabel('Power Consumption (KWhr)')
plt.ylabel('CO2 Emissions (Kt)')
plt.title("World's Power Consumption (KWhr) vs CO2 Emissions")
plt.show()

# Plotting the scatter plot for dataframe
plt.figure(5)
plt.scatter(df_forest_land_percent_1.loc['World'], df_co2_emm_1.loc['World'])
# Defining x and y label and title
plt.xlabel('Forest Land (%)')
plt.ylabel('CO2 Emissions (Kt)')
plt.title("World's Forest Land (%) vs CO2 Emissions")
plt.show()
