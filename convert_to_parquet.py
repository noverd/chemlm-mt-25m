import pandas as pd
import os
import sys

def convert_csv_to_parquet(csv_path="datset.csv", parquet_path="datset.parquet"):
    if not os.path.exists(csv_path):
        print(f"Ошибка: Файл {csv_path} не найден.")
        return

    print(f"Читаем {csv_path}...")
    try:
        # Читаем CSV. Если файл очень большой, можно читать чанками, 
        # но для 50МБ (как в вашем случае) pandas справится сразу.
        df = pd.read_csv(csv_path)
        
        print(f"Конвертируем в Parquet (engine='pyarrow')...")
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
        csv_size = os.path.getsize(csv_path) / (1024 * 1024)
        pq_size = os.path.getsize(parquet_path) / (1024 * 1024)
        
        print(f"SIZE CSV: {csv_size:.2f} MB")
        print(f"Размер Parquet: {pq_size:.2f} MB")
    except Exception as e:
        print(f"Произошла ошибка при конвертации: {e}")

if __name__ == "__main__":
    convert_csv_to_parquet()