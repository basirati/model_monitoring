import mlflow

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from prometheus_client import Gauge, Counter, Histogram, Summary, generate_latest

app = FastAPI(title="DemoModelAPI")

logged_model = 'model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

success_class_counter = Counter('success_class', 'Counting predictions for the success class')
normal_class_counter = Counter('normal_class', 'Counting predictions for the normal class')
items_gauge = Gauge('items_g', 'Gauge for value of items available in a store')
customers_gauge = Gauge('customers_g', 'Gauge for value of daily customers count in a store')
items_hist = Histogram('items_h', 'Hist for value of items available in a store',
                           buckets=[1397, 1507, 1631, 1718, 1773, 1844, 1939, 2046, 2185])
customers_hist = Histogram('customers_h', 'Hist for value of daily customers count in a store',
                            buckets=[440, 560, 650, 710, 780, 850, 925, 1020, 1140])

@app.get('/metrics')
def metrics():
    return HTMLResponse(content=generate_latest(), status_code=200)

@app.get('/predict')
def predict(items: int, customers: int):
    try:
        # performing the prediction by calling the model
        prediction = loaded_model.predict([[items, customers]])
        prediction = prediction[0].item()
        # exposing predicted class metrics
        if prediction:
            success_class_counter.inc()
        else:
            normal_class_counter.inc()
        # exposing input parameters metrics
        items_gauge.set(items)
        customers_gauge.set(customers)
        items_hist.observe(items)
        customers_hist.observe(customers)
        # return result in a json format
        return {'prediction': prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

@app.get('/error')
def oops():
    raise HTTPException(status_code=500, detail="Error endpoint for test!")
