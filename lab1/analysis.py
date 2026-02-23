import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression


def plot_wordcloud(df, path):
    word_freq = df.groupby("Топ 5")["Частота"].sum().to_dict()
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Хмара слів: Топ-терміни за тиждень моніторингу", fontsize=16, pad=20)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_dynamics(df, path):
    slots = df.drop_duplicates(subset=["День", "Час"])
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(slots)), slots["Сума частот"].values, marker="o", color="steelblue")
    labels = slots["День"] + "\n" + slots["Час"].apply(lambda x: x.split(":")[0])
    plt.xticks(range(len(slots)), labels, rotation=45, fontsize=8)
    plt.title("Динаміка сумарної частоти термінів (ранок, обід, вечір)", fontsize=14, pad=15)
    plt.xlabel("Часові проміжки протягом тижня", fontsize=12)
    plt.ylabel("Сума частот топ-5 термінів", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def forecast_and_plot(y, title, path):
    x = np.arange(len(y)).reshape(-1, 1)
    x_pred = np.arange(len(y) + 7).reshape(-1, 1)  # Прогноз на 7 днів вперед
    model = LinearRegression().fit(x, y)
    pred = model.predict(x_pred)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "o-", label="Реальні дані (поточний тиждень)", linewidth=2)
    plt.plot(x_pred, pred, "--", label="Лінія тренду та прогноз (+7 днів)", color="crimson", linewidth=2)
    plt.axvline(x=len(y) - 1, color="gray", linestyle=":", label="Межа прогнозу")
    plt.title(f"Аналіз та прогнозування: {title}", fontsize=14, pad=15)
    plt.xlabel("Дні моніторингу (0-6: історія, 7-13: прогноз)", fontsize=12)
    plt.ylabel("Частота згадувань", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    out_dir = "lab1_results"
    os.makedirs(out_dir, exist_ok=True)

    csv_file = "table1_monitoring.csv"
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)

    plot_wordcloud(df, f"{out_dir}/wordcloud.png")
    plot_dynamics(df, f"{out_dir}/dynamics.png")

    days = df["День"].unique()

    daily_total = df.drop_duplicates(subset=["День", "Час"]).groupby("День")["Сума частот"].sum().reindex(days).values
    forecast_and_plot(daily_total, "Загальна сума частот за день", f"{out_dir}/forecast_total.png")

    top3 = df["Топ 5"].value_counts().head(3).index.tolist()

    for term in top3:
        y_term = df[df["Топ 5"] == term].groupby("День")["Частота"].sum().reindex(days).fillna(0).values
        forecast_and_plot(y_term, f"Термін «{term}»", f"{out_dir}/forecast_{term}.png")



if __name__ == "__main__":
    main()