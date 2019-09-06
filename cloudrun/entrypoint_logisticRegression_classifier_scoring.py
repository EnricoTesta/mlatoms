from sklearn.linear_model import LogisticRegression
from predictor4docker import MyPredictor
from flask import Flask, request, jsonify
import os


app = Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    if not request.is_json():
        return "Invalid JSON request."

    # Fetch JSON payload. This should contain model-dir & prediction request dictionary.
    JSON_payload = request.get_json()

    # Reference model directory
    workdir = os.getcwd()

    # Copy model locally from GCS
    cmd = "gsutil rsync {} {}".format(JSON_payload["model-dir"], workdir)
    os.system(cmd)

    # instantiate predictor
    t = MyPredictor.from_path(workdir)
    return jsonify(t.predict(instances=JSON_payload["instances"], probabilities=JSON_payload["probabilities"],
                             info=JSON_payload["info"]))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
