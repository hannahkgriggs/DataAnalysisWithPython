import pandas as pd

def calculate_demographic_data(print_data=True):
    #read data from file
    df = pd.read_csv('adult.data.csv')

    #how many people of each race are represented in this dataset?
    race_count = df['race'].value_counts()

    #what is the average age of men?
    average_age_men = round(df[df['sex'] == 'Male']['age'].mean(), 1)

    #what is the percentage of people who have a Bachelor's degree?
    percentage_bachelors = round((df['education'] == 'Bachelors').mean() * 100, 1)

    #what percentage of people with advanced education (Bachelors, Masters, or Doctorate) make more than 50K?
    #what percentage of people without advanced education make more than 50K?

    #filter for advanced education
    higher_education_mask = df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])
    
    higher_education = df[higher_education_mask]
    lower_education = df[~higher_education_mask]

    #percentage with salary >50K
    higher_education_rich = round((higher_education['salary'] == '>50K').mean() * 100, 1)
    lower_education_rich = round((lower_education['salary'] == '>50K').mean() * 100, 1)

    #what is the minimum number of hours a person works per week?
    min_work_hours = df['hours-per-week'].min()

    #what percentage of the people who work the minimum number of hours per week have a salary of >50K?
    num_min_workers = df[df['hours-per-week'] == min_work_hours]
    rich_percentage = round((num_min_workers['salary'] == '>50K').mean() * 100, 1)

    #what country has the highest percentage of people that earn >50K and what is that percentage?
    country_total = df['native-country'].value_counts()
    country_rich = df[df['salary'] == '>50K']['native-country'].value_counts()

    highest_earning_country_percentage_series = (country_rich / country_total) * 100

    highest_earning_country = highest_earning_country_percentage_series.idxmax()
    highest_earning_country_percentage = round(highest_earning_country_percentage_series.max(), 1)

    #identify the most popular occupation for those who earn >50K in India.
    top_IN_occupation = df[(df['native-country'] == 'India') & (df['salary'] == '>50K')]['occupation'].value_counts().idxmax()


    if print_data:
        print("Number of each race:\n", race_count) 
        print("Average age of men:", average_age_men)
        print(f"Percentage with Bachelors degrees: {percentage_bachelors}%")
        print(f"Percentage with higher education that earn >50K: {higher_education_rich}%")
        print(f"Percentage without higher education that earn >50K: {lower_education_rich}%")
        print(f"Min work time: {min_work_hours} hours/week")
        print(f"Percentage of rich among those who work fewest hours: {rich_percentage}%")
        print("Country with highest percentage of rich:", highest_earning_country)
        print(f"Highest percentage of rich people in country: {highest_earning_country_percentage}%")
        print("Top occupations in India:", top_IN_occupation)

    return {
        'race_count': race_count,
        'average_age_men': average_age_men,
        'percentage_bachelors': percentage_bachelors,
        'higher_education_rich': higher_education_rich,
        'lower_education_rich': lower_education_rich,
        'min_work_hours': min_work_hours,
        'rich_percentage': rich_percentage,
        'highest_earning_country': highest_earning_country,
        'highest_earning_country_percentage':
        highest_earning_country_percentage,
        'top_IN_occupation': top_IN_occupation
    }
