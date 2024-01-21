# Importing required libraries for the program
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
from sklearn.cluster import KMeans
import sklearn.preprocessing as pp
import cluster_tools as ct
import errors as err


def file_to_df(filename):
    """
    This function will import the csv file into the program and read it in
    the World Bank format by skiping first rows and remove the columns
    not needed in the program.

    Parameters:
    - filename (str): The input DataFrame containing countries data.

    Returns:
    df: Dataframe containing the indicator data.
    """

    # Read the CSV file and skip the first four rows
    # Set the list of country names as the index of the dataframe
    df = pd.read_csv(filename, skiprows=4, index_col='Country Name')

    # Remove the unnecessary columns using slicing
    # Select data from last 30 years
    df = df.loc[:, '1991':'2020']

    # Return the dataframe
    return df


def latest_data(countries_lst, df_lst, df_name):
    """
    This function will get the list of dataframes and their names also
    the list of countries names of interest and then get the latest data
    from each dataframe for those countries and join into single dataframe.

    Parameters:
    - countries_lst (list): List of countries names.
    - df_lst (list): List of dataframes.
    - df_names (list): List of dataframe names.

    Returns:
    df_c: Dataframe containing the latest data for given countries.
    """
    # Define dataframe to put our data
    df_c = pd.DataFrame()
    i = 0

    # Using while loop to iterate through each dataframe
    while i < len(df_name):
        # Extract and append the latest data from dataframe using slicing
        df_c[df_name[i]] = df_lst[i].loc[countries_lst].\
            dropna(axis=1).iloc[:, -1]
        i += 1

    # Return resultant dataframe
    return df_c


def one_silhoutte(df_inp, n):
    """
    Calculates silhoutte score for n clusters
    """
    # Extract numerical values from the DataFrame
    xy = df_inp.values

    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)  # fit done on x,y pairs
    labels = kmeans.labels_

    # Calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


def poly(x, a, b, c, d, e):
    """
    Calulates polynominal
    """
    x = x - 1991
    f = a + b*x + c*x**2 + d*x**3 + e*x**4

    return f


# Ignore warnings
warnings.filterwarnings("ignore")

# Importing the data by calling the function file_to_df
df_co2 = file_to_df('CO2_EM.csv')
df_co2_pc = file_to_df('CO2_EM_PC.csv')
df_gdp_pc = file_to_df('GDP_PC.csv')
df_elec_pc = file_to_df('ELEC_CONS_PC.csv')
df_urb_pc = file_to_df('URB_POP.csv')
df_frst_pc = file_to_df('FRST_PC.csv')

# Make the list of data to give it to function
lst_of_dfs = [df_co2_pc, df_gdp_pc, df_elec_pc, df_urb_pc, df_frst_pc]

# Make the list of the dataframe names
lst_of_df_names = ['CO2 emission per capita (Kt)', 'GDP per capita ($)',
                   'Power consumption per capita (KWh)',
                   'Urban Population (%)', 'Forest Land (%)']

# Define the list of countries of interest
countries = ['China', 'United States', 'India', 'Russian Federation',
             'Japan', 'Germany', 'Canada']

# Convert the years from str to int
df_co2.columns = pd.to_numeric(df_co2.columns)

# Defining new figure to plot
plt.figure(figsize=(12, 6))

# Iterating through countries
for country in countries:
    # Plot the co2 emission of that country
    plt.plot(df_co2.columns, df_co2.loc[country], linewidth=2)

# Plot legend for the data
plt.legend(countries, fontsize=12)
# Plot the x and y axis labels and the title of the graph
plt.xlabel("Years", fontsize=12)
plt.ylabel("CO2 Emission (KT)", fontsize=12)
plt.title('CO2 emission data for top 7 countries for \
last 30 years', fontsize=16)
# Rotating X-axis labels
plt.xticks(rotation=45)
# Show the resultant plot
plt.show()

# Get the latest data for given countries by calling the function
df = latest_data(countries, lst_of_dfs, lst_of_df_names)

# Plot the correlation matrix plot for the data
ct.map_corr(df)

# Plot the scatter plot for the correlation matrix
pd.plotting.scatter_matrix(df, figsize=(14, 14))
# Show the resultant plot
plt.show()

# Defining new figure to plot
plt.figure(figsize=(12, 6))
# Plot the scatter plot for two series from the dataframe
plt.scatter(df["CO2 emission per capita (Kt)"],
            df["Power consumption per capita (KWh)"], 10, marker="o")
# Plot the x and y axis labels and the title of the graph
plt.xlabel("CO2 emission per capita (Kt)", fontsize=12)
plt.ylabel("Electric power consumption per capita (KWh)", fontsize=12)
plt.title('CO2 emission per capita (Kt) vs. Electric power \
consumption per capita (KWh) plot for top 7 countries')
# Show the resultant plot
plt.show()

# Calculate silhouette score for 2 to 6 clusters
for n in range(2, 7):
    score = one_silhoutte(df, n)
    print(f"The silhouette score for {n: 3d} is {score: 7.4f}")

# Define the scaler function
scaler = pp.RobustScaler()

# Make the cluster dataframe
df_clust = df[['CO2 emission per capita (Kt)', 'GDP per capita ($)']]
# Fit the cluster
scaler.fit(df_clust)
# Normalize the cluster
norm = scaler.transform(df_clust)

# Define kmean function
kmeans = KMeans(n_clusters=3, n_init=20)
# Fit the normalized cluster into the kmean
kmeans.fit(norm)
# Extract labels
labels = kmeans.labels_

# Extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

# Extract x and y values of data points
x = df_clust["CO2 emission per capita (Kt)"]
y = df_clust["GDP per capita ($)"]

# Defining new figure to plot
plt.figure(figsize=(12, 6))
# Plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o")
# Show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
# Plot the x and y axis labels and the title of the graph
plt.xlabel("CO2 emission per capita (Kt)")
plt.ylabel("GDP per capita ($)")
plt.title('CO2 emission per capita (Kt) vs. \
GDP per capita ($) with cluster centers')
# Show the resultant plot
plt.show()

# Define the forecasting variables using curve_fit function
param, covar = opt.curve_fit(poly, df_co2.columns, df_co2.loc['World'])
# Calculate sigma
sigma = np.sqrt(np.diag(covar))

# Deifne the years to forecast
year = np.arange(1991, 2031)
# Calculate the forecasted data
forecast = poly(year, *param)

# Calculate sigma error and calculate low and high
sigma = err.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma

# Defining new figure to plot
plt.figure(figsize=(12, 6))
# Plot the original data
plt.plot(df_co2.columns, df_co2.loc['World'], label="World")
# Plot the forecasted data
plt.plot(year, forecast, label="forecast")

# Plot uncertainty range using low and high
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
# Plot the x and y axis labels and the title of the graph
plt.xlabel("Year", fontsize=12)
plt.ylabel("CO2 Emission", fontsize=12)
plt.title('CO2 emission forecast for the world from 2020 to 2030', fontsize=16)
# Define legend
plt.legend()
# Show the resultant plot
plt.show()
