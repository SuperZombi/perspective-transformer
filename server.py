from flask import Flask, request, render_template, send_file, abort
import threading
import time
import os
import shutil
from PySimpleDB import DataBase
from perspective_transformation import Perspective


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

if os.path.exists("data"):
	shutil.rmtree("data")
if os.path.exists("data.bd"):
	os.remove("data.bd")

os.mkdir("data")
DB = DataBase("data.bd", unique="id")

@app.route("/")
def home():
	return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
	if request.method == "POST":
		file = request.files.get('image')
		if file:
			id = str(int(time.time()))
			os.mkdir(os.path.join("data", id))
			file.save(os.path.join("data", id, file.filename))
			DB.add(id=id, original_file=file.filename, status=0)
			threading.Thread(target=image_worker, kwargs={"id": id, "file": os.path.join("data", id, file.filename)}).start()
			return f"/result/{id}"

@app.route("/result/<id>/")
@app.route("/result/<id>/<file>")
def result(id, file=None):
	if not os.path.exists(os.path.join("data", id)):
		abort(404)
	if file:
		filepath = os.path.join("data", id, file)
		if not os.path.exists(filepath):
			abort(404)
		return send_file(filepath)

	data = DB.get(id)
	return render_template("result.html", **data)



def image_worker(id, file):
	row = DB.get(id)
	worker = Perspective(file)

	def filepath(name): return os.path.basename(name)

	row['binary'] = filepath(worker.preprocess())
	row['contour'] = filepath(worker.findContours())
	row['box'] = filepath(worker.findBox())
	row['transformed'] = filepath(worker.apply_transform())
	row['status'] = 1
	DB.save()


app.run(host='0.0.0.0', port=80)
