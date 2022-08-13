from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

labels_str = ['Bald Eagle', 'Snowy Owl']
model = load_model('classification_model.h5')
model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(100, 100))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    prob = int(np.round(p, 0))
    img_class = labels_str[prob]
    return img_class


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        pred_label = predict_label(img_path)
        return render_template("index.html", prediction=pred_label, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
