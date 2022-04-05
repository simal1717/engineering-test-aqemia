import http.client
import json


def make_prediction(url, port, endpoint, inputs):
    connection = http.client.HTTPConnection(f"{url}:{port}")
    headers = { "Content-Type": "application/json" }
    payload = json.dumps(inputs)
    connection.request("POST", endpoint, payload, headers)
    response = connection.getresponse()
    data = response.read()
    return data.decode("utf-8")