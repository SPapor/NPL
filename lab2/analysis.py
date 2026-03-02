import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_transformation(csv_file="lab2.csv"):
    df = pd.read_csv(csv_file)
    unique_days = df['День'].unique()

    midpoint = len(unique_days) // 2
    week1_days = set(unique_days[:midpoint])

    df['Період'] = df['День'].apply(lambda x: 'Тиждень 1' if x in week1_days else 'Тиждень 2')

    week1_top = df[df['Період'] == 'Тиждень 1'].groupby('Топ 5')['Частота'].sum().nlargest(5)
    week2_top = df[df['Період'] == 'Тиждень 2'].groupby('Топ 5')['Частота'].sum().nlargest(5)

    top_words_combined = set(week1_top.index) | set(week2_top.index)

    plot_data = df[df['Топ 5'].isin(top_words_combined)]
    grouped_data = plot_data.groupby(['Період', 'Топ 5'])['Частота'].sum().reset_index()

    plt.figure(figsize=(12, 7))
    sns.barplot(data=grouped_data, x='Топ 5', y='Частота', hue='Період', palette='viridis')

    plt.title('Трансформація ключових тем новин (Тиждень 1 vs Тиждень 2)', fontsize=16, pad=20)
    plt.xlabel('Ключові слова', fontsize=12)
    plt.ylabel('Сумарна частота згадувань', fontsize=12)
    plt.xticks(rotation=45, fontsize=11)
    plt.legend(title='Період')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('transformation_bar_chart.png', dpi=300)

if __name__ == "__main__":
    evaluate_transformation()