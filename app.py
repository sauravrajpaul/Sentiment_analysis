from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]
        sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
        return render_template("index.html", review=review, prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
