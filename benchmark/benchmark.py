import pandas as pd
from tabulate import tabulate
import time
import os
import csv

iterations = 100
path = 'benchmark/'
df = pd.read_csv('benchmark/Google_data_cleaned.csv')
benchmark_result = {'File format': [], 'Size': [], 'Write time [ms]': [], 'Read time [ms]': []}

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

def benchmark(format_name, save_func, load_func, path, key=None, mode=None):
    
    benchmark_result['File format'].append(format_name)
    start = time.time()
    if format_name != 'HDF5':
        for _ in range(iterations):
            save_func(path)
    else:
        for _ in range(iterations):
            save_func(path, key, mode)
    end = time.time()

    benchmark_result['Write time [ms]'].append((end - start) / iterations)
    
    start = time.time()
    for _ in range(iterations):
        load_func(path)
    end = time.time()

    benchmark_result['Read time [ms]'].append((end - start) / iterations)

if __name__ == '__main__':

    benchmark('CSV', df.to_csv,pd.read_csv, 'benchmark/data/data.csv')
    benchmark('Feather', df.to_feather, pd.read_feather, 'benchmark/data/data.feather')
    benchmark('HDF5', df.to_hdf, pd.read_hdf, 'benchmark/data/data.h5', key='df', mode='w')
    benchmark('JSON', df.to_json, pd.read_json, 'benchmark/data/data.json')
    benchmark('Parquet', df.to_parquet, pd.read_parquet, 'benchmark/data/data.parquet')

    for file_name in os.listdir('benchmark/data/'):
        file_path = os.path.join('benchmark/data/', file_name)
        file_size = os.path.getsize(file_path)
        human_readable_size = sizeof_fmt(file_size)
        benchmark_result['Size'].append(human_readable_size)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    table = [list(row) for row in zip(*benchmark_result.values())]
    headers = list(benchmark_result.keys())

    with open('benchmark/benchmark_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in table:
            writer.writerow(row)

    print(tabulate(table, headers=headers))
