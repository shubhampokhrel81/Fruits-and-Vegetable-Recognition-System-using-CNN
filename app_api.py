from flask import Flask, render_template, request, redirect, url_for, jsonify
from data import Articles
import os
from PIL import Image
from werkzeug.utils import secure_filename

import matplotlib.image as mpimg
import numpy as np
from PIL import *
from keras.models import load_model
import os.path
import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Model
import os, os.path
from keras import backend as K

app = Flask(__name__)
Articles = Articles()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/articles')
def articles():
    return render_template('articles.html', articles=Articles)


@app.route('/article/<string:id>')
def article(id):
    return render_template('article.html', articles=Articles, id=id)

def uploadPhoto():
    target = os.path.join(APP_ROOT, 'test_images/')
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        filename = file.filename	
        print(filename)
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    newDes = os.path.join('test_images/'+filename)
    print("newdes==",newDes)
    return newDes

def predict(newDes):

    train_categories = []
    for i in os.listdir("./data/merged/train"):
        train_categories.append(i)

    img = Image.open(newDes)
    original_img = np.array(img, dtype=np.uint8)

    if img.size[0] > img.size[1]:
        scale = 100 / img.size[1]
        new_h = int(img.size[1] * scale)
        new_w = int(img.size[0] * scale)
        new_size = (new_w, new_h)
    else:
        scale = 100 / img.size[0]
        new_h = int(img.size[1] * scale)
        new_w = int(img.size[0] * scale)
        new_size = (new_w, new_h)

    resized = img.resize(new_size)
    resized_img = np.array(resized, dtype=np.uint8)

    left = 10
    right = left + 100
    up = 0
    down = up + 100
    K.clear_session()
    model = load_model('model.h5')

    cropped = resized.crop((left, up, right, down))
    cropped_img = np.array(cropped, dtype=np.uint8)
    cropped_img = cropped_img / 255.0

    X = np.reshape(cropped_img, newshape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
    prediction_multi = model.predict(x=X)
    store = np.argmax(prediction_multi)
    
    print("tc",train_categories[store])
    K.clear_session()
    return train_categories[store]

def reset_model(model):
    for layer in model.layers:
        if hasattr(layer, 'init'):
            init = getattr(layer, 'init')
            new_weights = init(layer.get_weights()[0].shape).get_value()
            bias = shared_zeros(layer.get_weights()[1].shape).get_value()
            layer.set_weights([new_weights, bias])

@app.route('/api/uploads', methods=['POST'])
def apiUpload():
    
    newDes = uploadPhoto()

    prediction = predict(newDes)

    # APP_KEY = '2ac225cf90201e1e8fb696d3352f5f8a'	
    # APP_ID = '4a8e817b'
    # URL = 'https://api.edamam.com/search?q=banana'+'&app_id='+APP_ID+'&app_key='+APP_KEY+'&from=0&to=5'
    # data={}
    # print(URL)
    # headers = {"Accept": "application/json"}
    # myResponse = requests.get(URL,data=data)

    URL = 'http://2481d306.ngrok.io/app.php/api/v1/recipe/search?name='+prediction

    data={}
    print(URL)
    headers = {"Accept": "application/json"}
    myResponse = requests.get(URL,data=data)
    print(myResponse)
    # call get service with headers and params
    if(myResponse.ok):
        data = myResponse.json()
        print(data['data'])
        response = jsonify({
            'prediction': prediction,
            'recipes': data['data']
        })

    else:
        response = jsonify({
            'prediction': prediction,
            'recipes': ''
        })
        print('not ok')
    # call get service with headers and params
    # if(myResponse.ok):
    #     print(json.loads(myResponse.text)['hits'])
    #     response = jsonify({
    #         'firstPrediction': train_categories[top_pred[0][-1]],
    #         'recipes': json.loads(myResponse.text)['hits']
    #     })

    # else:
    #     print('not ok')
    #     response = jsonify({
    #         'firstPrediction': train_categories[top_pred[0][-1]],
    #         'recipes':'not found',
                
    #         })

    response.status_code = 200
    return response

@app.route('/upload', methods=['POST'])
def upload():

    newDes = uploadPhoto()

    prediction = predict(newDes)
    print(prediction)
    # APP_KEY = '2ac225cf90201e1e8fb696d3352f5f8a'	
    # APP_ID = '4a8e817b'
    #URL = 'https://api.edamam.com/search?q='+train_categories[top_pred[0][-1]]+'&app_id='+APP_ID+'&app_key='+APP_KEY+'&from=0&to=5'
    URL = 'http://192.168.100.192:8002/api/v1/recipe/search?name=banan'

    data={}
    print(URL)
    headers = {"Accept": "application/json"}
    myResponse = requests.get(URL,data=data)
    print(myResponse)
    # call get service with headers and params

    if(myResponse.ok):
        response = myResponse.json()
        print(response['data'])

    else:
        print('not ok')

    return render_template('about.html',results = results)
    #return (results, destination)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
