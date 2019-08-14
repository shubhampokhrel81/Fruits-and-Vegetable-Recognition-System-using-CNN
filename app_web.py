from flask import Flask, render_template, request, redirect, url_for
from data import Articles
import os

from werkzeug.utils import secure_filename

# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import *
from keras.models import load_model
import os.path

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


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'test_images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)	

    newDes = os.path.join('test_images/'+filename)
    
    train_categories = []
    # train_samples = []
    for i in os.listdir("./data/merged/train"):
        train_categories.append(i)
  
    # model.load_weights("finalmodel.hdf5")
    img = Image.open(newDes)
    original_img = np.array(img, dtype=np.uint8)
    # plt.imshow(original_img)

    if img.size[0] > img.size[1]:
        scale = 100 / img.size[1]
        new_h = int(img.size[1]*scale)
        new_w = int(img.size[0]*scale)
        new_size = (new_w, new_h)
    else:
        scale = 100 / img.size[0]
        new_h = int(img.size[1]*scale)
        new_w = int(img.size[0]*scale)
        new_size = (new_w, new_h)

    resized = img.resize(new_size)
    resized_img = np.array(resized, dtype=np.uint8)
    # plt.imshow(resized_img)

    left = 0
    right = left + 100
    up = 0
    down = up + 100
    model = load_model('model.h5')
    cropped = resized.crop((left, up, right, down))
    cropped_img = np.array(cropped, dtype=np.uint8)
    # plt.imshow(cropped_img)

    cropped_img = cropped_img / 255
    X = np.reshape(cropped_img, newshape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
    prediction_multi = model.predict(x=X)
    # print(np.argmax(prediction_multi))
    print("Fruit is : ", train_categories[np.argmax(prediction_multi)])
    fruit_name = train_categories[np.argmax(prediction_multi)]

    acc_sort_index = np.argsort(prediction_multi)
    top_pred = acc_sort_index[:, -6:]
    results =[train_categories[top_pred[0][-1]]]
    print(results)
    print("1st Prediction: ", train_categories[top_pred[0][-1]])
    print("2nd Prediction: ", train_categories[top_pred[0][-2]])
    print("3rd Prediction: ", train_categories[top_pred[0][-3]])


    return render_template('about.html',results = results)
    #return (results, destination)




if __name__ == '__main__':
    #app.run(debug=True)
    app.run(port=5000, debug=True, use_reloader=False)
