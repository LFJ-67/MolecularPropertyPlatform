from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import time
import uuid

from werkzeug.utils import secure_filename

from predictor import manager
from config import MODEL_CONFIG
from model import predict_new_smiles

from services.structure_service import smiles_to_base64, molecule_info
from services.batch_service import batch_predict_file, load_file

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok"
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "message": "文件过大，最大允许 10MB。"
    }), 413

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None:
        return jsonify({
            "success": False,
            "message": "请求数据不能为空。"
        })

    smiles = data.get("smiles", "").strip()
    target = data.get("target", "triplet")

    if smiles == "":
        return jsonify({
            "success": False,
            "message": "请输入SMILES。"
        })

    if target not in MODEL_CONFIG:
        return jsonify({
            "success": False,
            "message": "未知预测类型。"
        })

    model = manager.get_model(target)
    scaler = manager.get_scaler(target)

    start = time.perf_counter()

    try:

        pred = predict_new_smiles(
            smiles,
            model=model,
            scaler=scaler
        )

        if pred is None:
            return jsonify({
                "success": False,
                "message": "SMILES解析失败。"
            })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })

    elapsed = time.perf_counter() - start

    return jsonify({

        "success": True,

        "data": {

            "property": MODEL_CONFIG[target]["title"],

            "prediction": round(float(pred), 4),

            "unit": MODEL_CONFIG[target]["unit"],

            "time": round(elapsed, 3)

        }

    })



@app.route("/structure", methods=["POST"])
def structure():

    data = request.get_json()

    smiles = data.get("smiles", "").strip()

    img = smiles_to_base64(smiles)

    info = molecule_info(smiles)

    if img is None:

        return jsonify({

            "success": False,

            "message": "SMILES解析失败"

        })

    return jsonify({

        "success": True,

        "image": img,

        "info": info

    })



@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "No file."
        })

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "message": "No filename."
        })

    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "message": "仅支持 .csv、.xlsx、.xls 文件。"
        })

    filename = f"{uuid.uuid4().hex}{ext}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:
        df = load_file(filepath)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

    preview = df.head().to_dict(orient="records")

    return jsonify({
        "success": True,
        "filename": filename,
        "columns": list(df.columns),
        "preview": preview
    })


@app.route("/batch_predict", methods=["POST"])
def batch_predict():

    data = request.get_json()

    filename = data["filename"]

    target = data["target"]

    input_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )

    output_filename = f"result_{filename}"

    output_path = os.path.join(

        app.config["RESULT_FOLDER"],

        output_filename

    )

    try:

        batch_predict_file(

            input_path,

            output_path,

            target

        )

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        })

    return jsonify({

        "success": True,

        "download": output_filename

    })


@app.route("/download/<filename>")
def download_file(filename):

    return send_from_directory(

        app.config["RESULT_FOLDER"],

        filename,

        as_attachment=True

    )





if __name__ == "__main__":

    if not os.path.exists("templates"):
        os.makedirs("templates")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )







