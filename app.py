from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
knn = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Define the route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Define the route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    if text:
        vectorized_text = vectorizer.transform([text])
        prediction = knn.predict(vectorized_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", prediction=result)
    else:
        return render_template("index.html", prediction=None)


# Define the main block to run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
