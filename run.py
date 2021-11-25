"""
run.py is the entrypoint for USP docker container.
"""
import io
from PIL import Image
import segwscribb_dev
from flask import Flask, request, jsonify, copy_current_request_context
import numpy as np
import base64

app = Flask(__name__)

'''
Sending multiple data to flaskapp...
Sauce: https://jdhao.github.io/2020/04/12/build_webapi_with_flask_s2/

multiple_files = [
    ('image', ('test.jpg', open('test.jpg', 'rb'))),
    ('image', ('test.jpg', open('test.jpg', 'rb')))
]
# simplified form
# multiple_files = [
#     ('image', open('test.jpg', 'rb')),
#     ('image', open('test.jpg', 'rb'))
# ]
r = requests.post(url, files=multiple_files, data=data)
'''


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = request.json
        defect_arr = np.array(data['image'])
        scribble_arr = np.array(data['scribble'])

        # prediction = jsonify({"img": encoded_str})

        # for i, file in enumerate(files):
        #    if i == 0:
        #        defect_img = Image.open(file.stream)
        #    else:
        #        if file is not None:
        #            scribble = Image.open(file.stream)
        # prediction = segment(defect_img, scribble)
    # else:
    #    prediction = "ERROR"
    #prediction = segment(defect_arr, scribble_arr) #scribble_arr is not registering None

    #return {'prediction': prediction.tolist()}
    test = scribble_arr is None
    return {'prediction': str(test)}

def segment(defect_img, scribble):
    if scribble is None:
        mask = segwscribb_dev.segment(defect_img, scribble=None, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1,
                                      stepsize_con=1, stepsize_scr=0.5, maxIter=200)
    else:
        mask = segwscribb_dev.segment(defect_img, scribble=scribble, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1,
                                      stepsize_con=1, stepsize_scr=0.5, maxIter=200)
    return mask


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
