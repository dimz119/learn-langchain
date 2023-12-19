import json
import requests
"""
POST /my_runnable/invoke - invoke the runnable on a single input
POST /my_runnable/batch - invoke the runnable on a batch of inputs
POST /my_runnable/stream - invoke on a single input and stream the output
POST /my_runnable/stream_log - invoke on a single input and stream the output, including output of intermediate steps as it's generated
GET /my_runnable/input_schema - json schema for input to the runnable
GET /my_runnable/output_schema - json schema for output of the runnable
GET /my_runnable/config_schema - json schema for config of the runnable
""" 
response = requests.post(
    "http://localhost:8000/joke/invoke",
    json={'input': {'topic': 'cats'}}
)
print(response.json())
"""
{
    'output': {
        'content': "Sure, here's a cat-related joke for you:\n\nWhy don't cats play poker in the wild?\n\nToo many cheetahs!",
        'additional_kwargs': {},
        'type': 'ai',
        'example': False
    }, 
    'callback_events': [], 
    'metadata': {
        'run_id': '8ed5fa43-6e1f-4743-8310-77dbe52503c0'}}
"""

response = requests.get(
    "http://localhost:8000/joke/input_schema",
)
print(response.json())
"""
{'title': 'PromptInput', 'type': 'object', 'properties': {'topic': {'title': 'Topic', 'type': 'string'}}}
"""

response = requests.post(
    "http://localhost:8000/joke/stream",
    json={'input': {'topic': 'cats'}},
    stream=True
)

# Check if the request was successful
if response.status_code == 200:
    try:
        current_event = None
        event_data = []
        for line in response.iter_lines():
            if line:  # filter out keep-alive new lines
                # print(line.decode('utf-8'))
                line = line.decode('utf-8')
                """
                event: data
                data: {"content":"","additional_kwargs":{},"type":"AIMessageChunk","example":false}
                event: data
                data: {"content":"Sure","additional_kwargs":{},"type":"AIMessageChunk","example":false}
                event: data
                ...
                """
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    try:
                        parsed_data = json.loads(data)
                        """
                        {'run_id': '9d77c51b-6e0d-457f-a75e-0961f205cc8f'}
                        {'content': '', 'additional_kwargs': {}, 'type': 'AIMessageChunk', 'example': False}
                        {'content': 'Sure', 'additional_kwargs': {}, 'type': 'AIMessageChunk', 'example': False}
                        {'content': ',', 'additional_kwargs': {}, 'type': 'AIMessageChunk', 'example': False}
                        ...
                        """
                        event_data.append((current_event, parsed_data))
                        
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {data}")
    except KeyboardInterrupt:
        print("Streaming stopped by user.")
else:
    print(f"Failed to stream data: {response.status_code}")
