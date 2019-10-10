#!/bin/env python
# coding: utf-8

import os
from flask import Flask, request, jsonify, send_file
import numpy as np
#import keras
from tensorflow.keras.models import load_model
import tensorflow as tf


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.debug = True

model = load_model('my_model2.h5')
graph = tf.get_default_graph()

@app.route('/', methods=['POST'])
def predict():
    img_list = request.json.get('images')
    img_list = np.array(img_list, dtype=np.float32)
    global graph
    with graph.as_default():
        prediction = model.predict(img_list)
    response = {
        'data': {
            'prediction': np.argmax(prediction, axis=1).tolist()
        }
    }
    return jsonify(response), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(port=port)
    #app.run()