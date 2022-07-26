import os
import threading
import random
import requests
import time
import math

import numpy as np

endpoint = "http://localhost:8000/predict"

def call_predict(minutes: int, per_min_freq: int):
    """Send requests to the FastAPI app that serves the model.
    The requests are sent for the duration specified in minutes and frequency specified in per_min_freq.

    Parameters
    ----------
    minutes : int
        Duration for which the requests are sent.
    per_min_freq : int
        The number of requests per minute
    """
    count = minutes * per_min_freq
    sleep = 60 / per_min_freq
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
    try:
        # creating a load for five minutes with the same distribution as the training set
        mu_item, sigma_item = 1782, 300 # mean and standard deviation
        mu_customers, sigma_customers = 786, 265 # mean and standard deviation
        call_predict(5, 100)
        time.sleep(5)
        # creating a load for five minutes with a data drift
        mu_item, sigma_item = 300, 200 # mean and standard deviation
        mu_customers, sigma_customers = 1000, 265 # mean and standard deviation
        call_predict(5, 100)
    except Exception as e:
        print(e)
    
