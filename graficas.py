# plot_learning_curves.py
import json
import matplotlib.pyplot as plt
import glob

# === Global MAE por ronda ===
with open("results/global_metrics/mae_per_round.json") as f:
    global_metrics = json.load(f)
rounds = [m["round"] for m in global_metrics]
maes = [m["mae"] for m in global_metrics]
plt.plot(rounds, maes, marker='o')
plt.title("Global MAE per Round")
plt.xlabel("Round")
plt.ylabel("MAE")
plt.grid()
plt.savefig("results/global_metrics/global_mae_curve.png")
plt.show()

# === Curvas locales (loss y MAE) ===
for file in glob.glob("results/local_histories/*/*.json"):
    with open(file) as f:
        history = json.load(f)
    plt.figure()
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["mae"], label="MAE")
    plt.title(f"Local Training {file}")
    plt.legend()
    plt.savefig(file.replace(".json", ".png"))
    plt.close()
