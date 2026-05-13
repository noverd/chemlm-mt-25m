import pandas as pd
import matplotlib.pyplot as plt
import os

# скрипт для отрисовки результатов бенчмарков по эпохам
def plot_benchmarks(results_dir="benchmark_results"):
    if not os.path.exists(results_dir):
        print(f"папка {results_dir} не найдена. сначала запустите benchmark.py")
        return

    # 1. Сбор всех данных
    all_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    if not all_files:
        print("csv файлы не найдены")
        return

    data_frames = []
    for f in all_files:
        df = pd.read_csv(os.path.join(results_dir, f))
        data_frames.append(df)
    
    full_df = pd.concat(data_frames)
    
    subset = full_df[full_df['mode'] == 'Improved'].copy()
    

    subset['valid_num'] = subset['valid'].astype(int)
    
    stats = subset.groupby('epoch').agg({
        'tanimoto': 'mean',
        'valid_num': 'mean',
        'text_similarity': 'mean'
    }).sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = stats.index
    
    ax1.plot(epochs, stats['tanimoto'], marker='o', linewidth=3, label='Chemical (Tanimoto)')
    ax1.plot(epochs, stats['text_similarity'], marker='s', linestyle='--', label='Linguistic (TextSim)')
    ax1.set_title("Эволюция точности предсказаний", fontsize=14)
    ax1.set_xlabel("Эпоха", fontsize=12)
    ax1.set_ylabel("Точность (0.0 - 1.0)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    ax2.plot(epochs, stats['valid_num'], marker='D', color='green', linewidth=3)
    ax2.set_title("Процент химически валидных молекул (V)", fontsize=14)
    ax2.set_xlabel("Эпоха", fontsize=12)
    ax2.set_ylabel("Доля валидных", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    
    # сохраняем
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/benchmark_evolution.png"
    plt.savefig(save_path)
    print(f"графики эволюции сохранены в: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_benchmarks()
