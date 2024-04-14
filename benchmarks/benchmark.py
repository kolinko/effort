'''
pip install requests
pip install evalplus

'''

import json
import requests
import pdb
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl

pdb.set_trace()

samples = [
    dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"]))
    for task_id, problem in get_human_eval_plus().items()
]
write_jsonl("samples.jsonl", samples)
exit()

def send_post_query(url, data):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=None)
        return response.json()  # Assumes the server response is in JSON format
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

def send_get_query(url, params):
    try:
        response = requests.get(url, params=params, timeout=None)
        return response.json()  # Assumes the server response is in JSON format
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# URL of your server
url = "http://localhost:8080/q"

# Data for POST request
post_data = {
    "query": "your_query_here",
    "effort": 50,
    "numtokens": 100
}

# Parameters for GET request
get_params = {
    "query": "your_query_here",
    "effort": 50,
    "numtokens": 100
}

# Sending a POST request
print("Sending POST request...")
post_result = send_post_query(url, post_data)
print("Response from POST request:", post_result)

# Sending a GET request
print("Sending GET request...")
get_result = send_get_query(url, get_params)
print("Response from GET request:", get_result)
