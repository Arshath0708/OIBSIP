# Unemployment Analysis 
# Written by: Arshath Abdulla A

import pandas as pd
import matplotlib.pyplot as plt

file1 = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\OASIS\OIBSIP\Task2_Unemployment\Unemployment in India.csv"
file2 = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\OASIS\OIBSIP\Task2_Unemployment\Unemployment_Rate_upto_11_2020.csv"
def load_datasets():
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    print("First dataset shape:", df1.shape)
    print("Second dataset shape:", df2.shape)
    return df1, df2

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def plot_time_series(df):
    if "Date" in df.columns and "Estimated Unemployment Rate (%)" in df.columns:
        plt.figure(figsize=(10,5))
        plt.plot(df["Date"], df["Estimated Unemployment Rate (%)"])
        plt.xlabel("Date")
        plt.ylabel("Unemployment Rate (%)")
        plt.title("Unemployment Rate Over Time (COVID Period)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_state_wise(df):
    if "Region" in df.columns and "Estimated Unemployment Rate (%)" in df.columns:
        avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values()
        plt.figure(figsize=(10,7))
        plt.barh(avg.index, avg.values)
        plt.xlabel("Average Unemployment Rate")
        plt.title("State-wise Average Unemployment Rate")
        plt.tight_layout()
        plt.show()
        return avg

def show_insights(avg_data):
    print("\nTop 5 Highest Unemployment States:")
    print(avg_data.sort_values(ascending=False).head())
    print("\nTop 5 Lowest Unemployment States:")
    print(avg_data.head())

if __name__ == "__main__":
    df_main, df_covid = load_datasets()
    df_main = clean_columns(df_main)
    df_covid = clean_columns(df_covid)
    plot_time_series(df_covid)
    avg_data = plot_state_wise(df_main)
    show_insights(avg_data)
