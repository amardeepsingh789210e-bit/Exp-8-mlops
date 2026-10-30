from flask import Flask, jsonify, request
app=Flask(__name__)
model=joblib.load("model.pkl")
@app.route("/")
def home():
    return jsonify({"message":"Welcome to the Iris Species Prediction API"})
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    s_l=data["sepal_length"]
    s_w=data["sepal_width"]
    p_l=data["petal_length"]
    p_w=data["petal_width"]
    features = [s_l, s_w, p_l, p_w]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})
if __name__=="__main__":
    app.run(host='0.0.0.0', port=5001)