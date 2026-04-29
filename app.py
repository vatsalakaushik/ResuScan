import os
import requests

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
download_file("https://drive.google.com/file/uc?export=download&id=1CizDU9l-Yazrcai5Ja-b3tq4vk4N1J8v", "tfidf.pkl")
download_file("https://drive.google.com/file/uc?export=download&id=1DhHBsbLz72ZRd2gkmKu9ursCBWJLyx1x", "clf.pkl")
download_file("https://drive.google.com/file/uc?export=download&id=1yBYV1xnvwaUSlpwPk6bEOMXdxtnD89Es", "encoder.pkl")


from flask import Flask, render_template, request, redirect, session
import pickle
import PyPDF2
import docx
import csv
import os

app = Flask(__name__)
app.secret_key = "resuscan_secret_key"

# -------- LOAD MODELS --------
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("clf.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# -------- ENSURE CSV EXISTS --------
if not os.path.exists("users.csv"):
    with open("users.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "phone", "email", "password"])

# -------- TEXT EXTRACTION --------
def extract_text(file):
    filename = file.filename

    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() or "" for page in reader.pages])

    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])

    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")

    return ""

# -------- PREDICTION --------
def make_prediction(text):
    transformed = tfidf.transform([text]).toarray()
    pred = model.predict(transformed)[0]
    return encoder.inverse_transform([pred])[0]

# -------- ROUTES --------
@app.route('/')
def home():
    return render_template('index.html', user=session.get("user"))

@app.route('/app')
def app_page():
    if "user" not in session:
        return redirect("/")
    return render_template('app.html', user=session.get("user"))

# -------- SIGNUP --------
@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("name")
    phone = request.form.get("phone")
    email = request.form.get("email")
    password = request.form.get("password")

    with open("users.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, phone, email, password])

    session["user"] = name
    session["email"] = email
    return redirect("/app")

# -------- LOGIN --------
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    with open("users.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) < 4:
                continue

            if row[2] == email and row[3] == password:
                session["user"] = row[0]
                session["email"] = row[2]
                return redirect("/app")

    return "Invalid credentials"

# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# -------- PREDICT --------
@app.route("/predict", methods=["POST"])
def predict_route():
    if "user" not in session:
        return redirect("/")

    file = request.files["resume"]

    if file.filename == "":
        return redirect("/app")

    text = extract_text(file)
    result = make_prediction(text)

    return render_template("app.html", result=result, user=session.get("user"))



# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)