from flask import Flask, render_template, flash, redirect, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy import desc
import logging
import os
import numpy as np
import pickle
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SECRET_KEY'] = 'SECRET KEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(basedir, 'cancer.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)


class Cancer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    thickness = db.Column(db.Integer)
    size = db.Column(db.Integer)
    shape = db.Column(db.Integer)
    adhesion = db.Column(db.Integer)
    epithelial = db.Column(db.Integer)
    chromatin = db.Column(db.Integer)
    nucleoli = db.Column(db.Integer)
    mitoses = db.Column(db.Integer)
    predict = db.Column(db.Integer)
    result = db.Column(db.Integer)

    def __init__(self,
                 name,
                 thickness,
                 size,
                 shape,
                 adhesion,
                 epithelial,
                 chromatin,
                 nucleoli,
                 mitoses,
                 predict,
                 result):

        self.name = name
        self.thickness = thickness
        self.size = size
        self.shape = shape
        self.adhesion = adhesion
        self.epithelial = epithelial
        self.chromatin = chromatin
        self.nucleoli = nucleoli
        self.mitoses = mitoses
        self.predict = predict
        self.result = result

# Product Schema


class ProductSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'thickness', 'size', 'shape', 'adhesion',
                  'epithelial', 'chromatin', 'nucleoli', 'mitoses', 'predict', 'result')


# Init schema
product_schema = ProductSchema()  # strict=True
products_schema = ProductSchema(many=True)  # strict=True


@app.route('/')
def deneme():
    return render_template('index.html')


@app.route('/learn', methods=['GET'])
def learn():
    return render_template('learn.html')


# else:
    #    return render_template('ax.html')
@app.route('/denemeler')
def denemeler():

    cancers = Cancer.query.order_by(desc(Cancer.id)).limit(5).all()
    return render_template('deneme2.html', cancers=cancers)


@app.route('/sonuclar', methods=['GET', 'POST'])
def sonuclar():

    if request.method == "POST":
        model = model = pickle.load(open('static/svm_tuned_model.sav', 'rb'))
        name = request.form.get('name')
        feature_1 = request.form['thickness']
        feature_2 = request.form['size']
        feature_3 = request.form['shape']
        feature_4 = request.form['adhesion']
        feature_5 = request.form['epithelial']
        feature_6 = request.form['chromatin']
        feature_7 = request.form['nucleoli']
        feature_8 = request.form['mitoses']
        result = request.form['result']
        print(result)

        input_array = np.array([float(feature_1), float(feature_2), float(feature_3), float(feature_4), float(feature_5), float(feature_6),
                                float(feature_7), float(feature_8)])
        input_array = input_array.reshape(1, -1)

        y_pred = model.predict(input_array)

        new_cancer = Cancer(name, int(feature_1), int(feature_2), int(feature_3), int(feature_4), int(
            feature_5), int(feature_6), int(feature_7), int(feature_8), int(y_pred[0]), int(result))
        db.session.add(new_cancer)

        db.session.commit()

        cancers = Cancer.query.order_by(desc(Cancer.id)).limit(5).all()
        if y_pred == 0:
            return render_template('sonuclar.html', sonuc_iyi="I Come With Good News", sonuc_kotu="", cancers=cancers)
        if y_pred == 1:
            return render_template('sonuclar.html', sonuc_kotu="I'll give some sad news, unfortunately the tumor is malignant", sonuc_iyi="", cancers=cancers)
    else:
        cancers = Cancer.query.order_by(desc(Cancer.id)).limit(5).all()
        return render_template('sonuclar.html', cancers=cancers)



if __name__ == '__main__':
    app.run(debug=True, port=1111)
