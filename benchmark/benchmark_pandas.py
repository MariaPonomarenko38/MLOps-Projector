import pandas as pd
import timeit
import time

iterations = 1000

def benchmark(func, *args, **kwargs):
    start = time.time()
    for i in range(iterations):
        func(*args, **kwargs)
    end = time.time()
    return (end - start) / iterations


df = pd.read_csv('benchmark/Google_data_cleaned.csv')

save_to_feather = benchmark(df.to_feather, 'benchmark/data.feather')
save_to_parquet = benchmark(df.to_parquet, 'benchmark/data.parquet')
save_to_csv = benchmark(df.to_csv, 'benchmark/data.csv')
save_to_json = benchmark(df.to_json, 'benchmark/data.json')

load_feather = benchmark(pd.read_feather, 'benchmark/data.feather')
load_parquet = benchmark(pd.read_parquet, 'benchmark/data.parquet')
load_csv = benchmark(pd.read_csv, 'benchmark/data.csv')
load_json = benchmark(pd.read_json, 'benchmark/data.json')
