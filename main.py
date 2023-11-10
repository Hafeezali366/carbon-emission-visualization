# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:44:47 2023

@author: Hafeez Ali
"""

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt


def filter_df(dataframe, country_name):
    """
    This function will filter the input DataFrame to include only data for a 
    specific country and reset its index and select the required values.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing countries data.
    - country_name (str): The name of the country to filter the data for.

    Returns:
    dataframe (pd.DataFrame): A new DataFrame containing only the data for
                              the specified country, with the index reset.
    """
    # Select only the selected country data from the dataset
    dataframe = dataframe[dataframe["country_name"] == country_name]
    # Reset the index values
    dataframe = dataframe.reset_index(drop=True)
    # Drop columns from the data to only include required values
    dataframe = dataframe.drop(['country_code', 'country_name'], axis=1)

    return dataframe


# Import the csv data using pandas read_csv method
df = pd.read_csv("co2_emissions_kt_by_country.csv")
# Call the filter_df funcation to get the data for specofied countries
df_uk = filter_df(df, "United Kingdom")
df_fr = filter_df(df, "France")
df_it = filter_df(df, "Italy")

# Ploting first figure
plt.figure(1)
# Plot the line plot for three countries using plot function in pyplot
plt.plot(df_uk['year'], df_uk['value'], label='United Kingdom')
plt.plot(df_fr['year'], df_fr['value'], label='France')
plt.plot(df_it['year'], df_it['value'], label='Italy')
# Adding x and y labels and title to the plot
plt.xlabel('Year')
plt.ylabel('CO2 Emission Level')
plt.title('Carbon Emission Levels from 1960 to 2019')
# Setting x-axis limit of the plot
plt.xlim(min(df['year']), max(df['year']))
# Add legend to the plot
plt.legend()
# Save the plot figure
plt.savefig('line_plot.png', bbox_inches='tight')
plt.show()

# Ploting second figure
plt.figure(2)
# Plot the scatter plot for three countries using scatter function in pyplot
plt.scatter(df_uk['year'], df_uk['value'], label='United Kingdom')
plt.scatter(df_fr['year'], df_fr['value'], label='France')
plt.scatter(df_it['year'], df_it['value'], label='Italy')
# Adding x and y labels and title to the plot
plt.xlabel('Year')
plt.ylabel('CO2 Emission Level')
plt.title('Carbon Emission Levels from 1960 to 2019')
# Setting x-axis limit of the plot
plt.xlim(min(df['year']), max(df['year']))
# Add legend to the plot
plt.legend()
# Save the plot figure
plt.savefig('scatter_plot.png', bbox_inches='tight')
plt.show()

# Ploting third figure
plt.figure(3)
# Plot the bar plot for three countries using bar function in pyplot
plt.bar(df_uk['year'], df_uk['value'], label='United Kingdom')
plt.bar(df_fr['year'], df_fr['value'], label='France')
plt.bar(df_it['year'], df_it['value'], label='Italy')
# Adding x and y labels and title to the plot
plt.xlabel('Year')
plt.ylabel('CO2 Emission Level')
plt.title('Carbon Emission Levels from 1960 to 2019')
# Setting x-axis limit of the plot
plt.xlim(min(df['year']), max(df['year']))
# Add legend to the plot
plt.legend()
# Save the plot figure
plt.savefig('bar_plot.png', bbox_inches='tight')
plt.show()
