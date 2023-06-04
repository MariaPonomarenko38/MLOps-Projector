import requests
import statistics
import time
import numpy as np

url = "http://localhost:8080/predict"  
batch_folders = ["batch_size_9", "batch_size_30", "batch_size_60"]

results = []
for folder in batch_folders:
    
    start_time = time.time()  
    response = requests.post(url, json={"bucket_name": "benchmark", "folder_name": folder})

    end_time = time.time() 
    processing_time = end_time - start_time

    results.append(processing_time)
    print(folder, processing_time)

median_time = statistics.median(results)
percentile_99 = np.percentile(results, 99)
percentile_90 = np.percentile(results, 90)
average_time = statistics.mean(results)
min_time = min(results)
max_time = max(results)

print("Results:")
print(f"Median time: {median_time:.3f} seconds")
print(f"99th percentile: {percentile_99:.3f} seconds")
print(f"90th percentile: {percentile_90:.3f} seconds")
print(f"Average time: {average_time:.3f} seconds")
print(f"Minimum time: {min_time:.3f} seconds")
print(f"Maximum time: {max_time:.3f} seconds")