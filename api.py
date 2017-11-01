from flask import Flask, request, jsonify
import json
import numpy as np
from utils import think, classify
app = Flask(__name__)

# probability threshold
# ERROR_THRESHOLD = 0.2

# load our calculated synapse values
# synapse_file = 'synapses.json' 
synapse_file = 'synapses_specific.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])
    words = synapse['words']
    classes = synapse['classes']

@app.route('/', methods = ['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return 'Hello, World!'

    if request.method == 'POST':
        content = request.get_json()
        results = classify(content['text'], synapse_0, synapse_1, words, classes)
        return jsonify({'sentence':content['text'], 'results': results })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1234, debug=True)