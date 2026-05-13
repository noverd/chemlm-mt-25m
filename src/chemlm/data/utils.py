import pandas as pd
import os


# загрузка данных для обучения
def load_mt_data(path: str | None = None) -> list[tuple[str, str]]:
    if path is None:
        path = "datset.parquet" if os.path.exists("datset.parquet") else "datset.csv"

    print(f"reading: {path}")
    try:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception as e:
        print(f"error while reading: {e}")
        return []

    raw: list[tuple[str, str]] = []
    # лимит длины чтоб не ломать эмбеддинги
    max_l = 115

    for row in df.itertuples(index=False):
        # input - реагенты, target - продукт
        r, p = str(row.input).strip(), str(row.target).strip()

        if not r or not p or r == 'nan' or p == 'nan':
            continue

        if len(r) > max_l or len(p) > max_l:
            continue

        # делаем две задачи сразу
        raw.append((f"[SYNTHESIS]{r}", p))
        raw.append((f"[RETRO]{p}", r))

    print(f"готово, загрузили {len(raw)} примеров")
    return raw
