import os
import threading
import random
import requests
import time

min_items = 932
max_items = 2667
min_customers = 10
max_customers = 1560

def call_predict(count: int, sleep: int):
    endpoint = "http://localhost:8000/predict"
    for index in range(0, count):
        try:
            items = random.randint(min_items, max_items)
            customers = random.randint(min_customers, max_customers)
            url = endpoint + f"?items={items}&customers={customers}"
            requests.get(url, timeout=1)
            time.sleep(sleep)
        except Exception as e:
            print(f'An exception at generator {os.path.basename(__file__)}: {e}')

if __name__ == '__main__':
    call_predict(500, 0.1)
