import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

#add an overweight column to the data
# Calculate BMI: weight (kg) / (height (m) ** 2)
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

#normalize data by making 0 always good and 1 always bad
# If cholesterol or gluc is 1 -> 0; if > 1 -> 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


#draw the Categorical Plot in the draw_cat_plot function
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    #group and reformat the data in df_cat to split it by cardio. Show the counts of each feature.
    # Group by cardio, variable, and value, then aggregate by size/count.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    #convert the data into long format and create a chart using sns.catplot()
    g = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )

    #get the figure for the output and store it in the fig variable
    fig = g.fig

    # 9. Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


#draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 11. Clean the data in the df_heat variable
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    #calculate the correlation matrix
    corr = df_heat.corr()

    #generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    #set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    #plot the correlation matrix using sns.heatmap()
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    #do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
