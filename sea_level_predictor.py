import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    #read data from file
    df = pd.read_csv('epa-sea-level.csv')

    #create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], color='blue', label='Original Data')

    #create first line of best fit (using all data through 2050)
    res_all = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    
    #generate years array from start year (1880) to 2050
    years_all = pd.Series([i for i in range(1880, 2051)])
    sea_levels_all = res_all.intercept + res_all.slope * years_all
    
    ax.plot(years_all, sea_levels_all, 'r', label='Best Fit Line (1880-2050)')

    #create second line of best fit (using data from year 2000 through most recent year)
    df_recent = df[df['Year'] >= 2000]
    res_recent = linregress(df_recent['Year'], df_recent['CSIRO Adjusted Sea Level'])
    
    #generate years array from 2000 to 2050
    years_recent = pd.Series([i for i in range(2000, 2051)])
    sea_levels_recent = res_recent.intercept + res_recent.slope * years_recent
    
    ax.plot(years_recent, sea_levels_recent, 'green', label='Best Fit Line (2000-2050)')

    #add labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level (inches)')
    ax.set_title('Rise in Sea Level')

    #save plot and return data for testing
    plt.savefig('sea_level_plot.png')
    return ax.get_figure()
