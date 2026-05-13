import pandas as pd
import matplotlib.pyplot as plt
import os

# скрипт для отрисовки графиков обучения
def plot_results(csv_path="logs/train_metrics.csv"):
    if not os.path.exists(csv_path):
        print("лог-файл не найден")
        return

    # читаем данные
    df = pd.read_csv(csv_path, comment='#')
    
    # создаем картинку с двумя графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. график Loss (сглаженный)
    window = 50
    df['loss_smooth'] = df['loss'].rolling(window=window).mean()
    
    ax1.plot(df.index, df['loss'], alpha=0.3, color='blue', label='loss (батч)')
    ax1.plot(df.index, df['loss_smooth'], color='blue', linewidth=2, label=f'loss (среднее по {window})')
    ax1.set_title("динамика ошибки (Loss)")
    ax1.set_xlabel("шаг обучения (батчи)")
    ax1.set_ylabel("значение Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. график Learning Rate
    ax2.plot(df.index, df['lr'], color='orange', linewidth=2)
    ax2.set_title("изменение скорости обучения (Learning Rate)")
    ax2.set_xlabel("шаг обучения (батчи)")
    ax2.set_ylabel("LR")
    ax2.set_yscale('log') # логарифмическая шкала для наглядности
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # сохраняем в новую папку
    save_path = "plots/training_plots.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    print(f"графики сохранены в: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_results()
