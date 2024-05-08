from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/predict",methods=["POST"])
def predict():
    Credit_Score=[float(x)for x in request.form.values()]
    final_credit=[np.array(Credit_Score)]
    output=model.predict(final_credit)
    return render_template("res.html",prediction_text="The credit status is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)