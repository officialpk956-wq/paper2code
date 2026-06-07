import requests

# Test Tutor Ask
resp = requests.post("http://127.0.0.1:8000/api/tutor/ask", json={
    "session_id": "test_session",
    "context_type": "module",
    "context_data": {
        "paper_title": "Test Paper",
        "layer_name": "Conv1",
        "module_type": "Conv2d",
        "explanation": "A standard convolution layer.",
        "flops_context": {"flops": "high"}
    },
    "query": "Why use this layer?"
})
print("ASK:", resp.json())

# Test Tutor Quiz
resp2 = requests.post("http://127.0.0.1:8000/api/tutor/quiz", json={
    "module_data": {
        "paper_title": "Test Paper",
        "layer_name": "Conv1",
        "module_type": "Conv2d",
        "explanation": "A standard convolution layer.",
    }
})
print("QUIZ:", resp2.json())

# Test Learning Path
resp3 = requests.get("http://127.0.0.1:8000/api/tutor/learning-path")
print("PATH:", resp3.json())
