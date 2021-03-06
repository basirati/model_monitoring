import mlflow

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from prometheus_client import Gauge, Counter, Histogram, Summary, generate_latest

app = FastAPI(title="DemoModelAPI")

# loading model
logged_model_path = 'model'
loaded_model = mlflow.pyfunc.load_model(logged_model_path)

# counter for success class
success_class_counter = Counter('success_class', 'Counting predictions for the success class')
# counter for normal class
normal_class_counter = Counter('normal_class', 'Counting predictions for the normal class')
# guage for items count
items_gauge = Gauge('items_g', 'Gauge for value of items available in a store')
# TODO: add a guage for customers
# TODO: add a histogram for items
# TODO: add a histogram for customers


@app.get('/metrics')
def metrics():
    """Expose all generated metrics for Prometheus
    """
    return HTMLResponse(content=generate_latest(), status_code=200)

@app.get('/predict')
def predict(items: int, customers: int):
    """Predicts success of a store based on its items count and customers count
    """
    try:
        # performing the prediction by calling the model
        prediction = loaded_model.predict([[items, customers]])
        prediction = prediction[0].item()
        
        # intrumentation secion:
        # exposing predicted class metrics
        if prediction:
            success_class_counter.inc()
        else:
            normal_class_counter.inc()
        # exposing input data metrics
        items_gauge.set(items)
        
        # TODO: expose added metrics

        
        # return result in a json format
        return {'prediction': prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")
