import requests
import statistics
import time
import numpy as np

url = "http://localhost:8080/predict"  
batch_folders = ["batch_size_9", "batch_size_30", "batch_size_60"]

results = []
with open("results.txt", "w") as file:
    for folder in batch_folders:
        
        start_time = time.time()  
        response = requests.post(url, json={"bucket_name": "benchmark", "folder_name": folder})

        end_time = time.time() 
        processing_time = end_time - start_time

        results.append(processing_time)
        file.write(f"Folder: {folder}, Processing Time: {processing_time:.3f} seconds\n")

    median_time = statistics.median(results)
    average_time = statistics.mean(results)
    min_time = min(results)
    max_time = max(results)

    file.write("Results:\n")
    file.write(f"Median time: {median_time:.3f} seconds\n")
    file.write(f"Average time: {average_time:.3f} seconds\n")
    file.write(f"Minimum time: {min_time:.3f} seconds\n")
    file.write(f"Maximum time: {max_time:.3f} seconds\n")