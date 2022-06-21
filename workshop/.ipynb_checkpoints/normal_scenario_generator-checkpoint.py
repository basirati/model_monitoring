import os
import threading
import random
import requests
import time
import math

import numpy as np

mu_item, sigma_item = 1782, 300 # mean and standard deviation
mu_customers, sigma_customers = 786, 265 # mean and standard deviation

def call_predict(count: int, sleep: int):
    endpoint = "http://localhost:8000/predict"
    items_dist = np.random.normal(mu_item, sigma_item, count)
    customers_dist = np.random.normal(mu_customers, sigma_customers, count)
    for items, customers in zip(items_dist, customers_dist):
        try:
            items_int = math.ceil(items)
            customers_int = math.ceil(customers)
            url = endpoint + f"?items={items_int}&customers={customers_int}"
            requests.get(url, timeout=1)
            time.sleep(sleep)
        except Exception as e:
            print(f'An exception at generator {os.path.basename(__file__)}: {e}')

if __name__ == '__main__':
    mins = 5
    for index in range(mins):
        call_predict(600, 0.1)
