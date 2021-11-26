"""
run.py is the entrypoint for USP docker container.
"""
import segwscribb_dev
from flask import Flask, request, jsonify, copy_current_request_context
import numpy as np
import base64
#from utils import rand_perspective_transform, rand_motion_blur

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
        minLabel = int(data['minLabels'])
        nChannel = int(data['nChannel'])
        lr = float(data['lr'])
        stepsize_sim = float(data['stepsize_sim'])
        stepsize_con = float(data['stepsize_con'])
        stepsize_scr = float(data['stepsize_scr'])
        maxIter = int(data['maxIter'])

        if data['scribble'] is None:
            scribble_arr = None
        else:
            scribble_arr = np.array(data['scribble'])

        prediction = segwscribb_dev.segment(im=defect_arr, scribble=scribble_arr, minLabels=minLabel, nChannel=nChannel,
                                            lr=lr, stepsize_sim=stepsize_sim, stepsize_con=stepsize_con,
                                            stepsize_scr=stepsize_scr, maxIter=maxIter)

    return {'prediction': prediction.tolist()}


if __name__ == "__main__":
    app.run(host='0.0.0.0')
