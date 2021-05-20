import re
import base64
import numpy as np
import tensorflow.keras as keras
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img


app = Flask(__name__)

model_file = './model/output/model2.h5'
global model
model = keras.models.load_model(model_file)
print(model.predict_classes(np.zeros((1, 28, 28, 1))))
@app.route('/')
def index():

    response = {}
    return render_template("index.html", **response)  # 如果没有使用 redis 统计访问次数功能，请使用index.html

@app.route('/predict/', methods=['Get', 'POST'])
def preditc():

    parseImage(request.get_data())
    img = img_to_array(load_img('temp.png', target_size=(28, 28), color_mode="grayscale")) / 255.
    img = np.expand_dims(img, axis=0)
    code = model.predict_classes(img)[0]
    response = {}
    response['code'] = int(code)
    print(response)
    return jsonify(response)


def parseImage(imgData):
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./temp.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=3335)