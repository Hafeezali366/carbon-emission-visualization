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
    
    # Return resultant DataFrame
    return dataframe


def lineplot(df, headers):
    """
    This function will plot the line plot for the given DataFrame with
    appropriate labels, title, legend and save the image.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing countries data.
    - headers (list): List with the name of countries to plot the graph.
    """
    # Intitialize the figure
    plt.figure()
    
    # Iterate the headers list to plot values for each country
    for head in headers:
        plt.plot(df['year'], df[head], label=head)
    
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
    # Show the plot
    plt.show()


def scatterplot(df, headers):
    """
    This function will plot the scatter plot for the given DataFrame with
    appropriate labels, title, legend and save the image.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing countries data.
    - headers (list): List with the name of countries to plot the graph.
    """
    # Intitialize the figure
    plt.figure()
    
    # Iterate the headers list to plot values for each country
    for head in headers:
        plt.scatter(df['year'], df[head], label=head)
    
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
    # Show the plot
    plt.show()


def barplot(df, headers):
    """
    This function will plot the bar plot for the given DataFrame with
    appropriate labels, title, legend and save the image.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing countries data.
    - headers (list): List with the name of countries to plot the graph.
    """
    # Intitialize the figure
    plt.figure()
    
    # Iterate the headers list to plot values for each country
    for head in headers:
        plt.bar(df['year'], df[head], label=head)
    
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
    # Show the plot
    plt.show()


# Import the csv data using pandas read_csv method
dataset = pd.read_csv("co2_emissions_kt_by_country.csv")

# Call the filter_df funcation to get the data for specofied countries
df_uk = filter_df(dataset, "United Kingdom")
df_fr = filter_df(dataset, "France")
df_it = filter_df(dataset, "Italy")

# Combine the data into one DataFrame
df = pd.DataFrame({
    'year': df_uk['year'],
    'United Kingdom': df_uk['value'],
    'France': df_fr['value'],
    'Italy': df_it['value']
})

# Define list of countries to plot
header = ["United Kingdom", "France", "Italy"]

# Call all three functions to plot the three graphs
lineplot(df, header)
scatterplot(df, header)
barplot(df, header)
