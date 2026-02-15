import os
import base64
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Silence TensorFlow INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flask setup
app = Flask(__name__)
CORS(app)

# Load model and warm up
model = tf.keras.models.load_model("best_model.keras")
model(tf.zeros((1,224,224,3)))

# Split model into feature extractor + head
base_model       = model.get_layer("efficientnetb0")
gap_layer        = model.get_layer("global_average_pooling2d")
dropout_layer    = model.get_layer("dropout")
dense_layer      = model.get_layer("dense")

# Preprocess image to (224,224,3) float32 in [0,255]
def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    return img.astype("float32")

# Classic Grad–CAM
def make_gradcam_overlay(img_bgr, alpha=0.4):
    # 1) Preprocess & batch
    proc  = preprocess(img_bgr)              # (224,224,3)
    batch = tf.convert_to_tensor(proc[None,...])  # (1,224,224,3)

    # 2) Forward + record gradients
    with tf.GradientTape() as tape:
        # a) conv feature maps
        conv_maps = base_model(batch, training=False)  # (1,7,7,1280)
        tape.watch(conv_maps)
        # b) head
        x = gap_layer(conv_maps)               # (1,1280)
        x = dropout_layer(x, training=False)   # (1,1280)
        preds = dense_layer(x)                 # (1,1) sigmoid
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    # 3) gradients & pooled weights
    grads = tape.gradient(loss, conv_maps)      # (1,7,7,1280)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))  # (1280,)

    # 4) build CAM
    conv_maps = conv_maps[0]                    # (7,7,1280)
    cam = tf.reduce_sum(conv_maps * pooled_grads[tf.newaxis,tf.newaxis,:], axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)     # normalize
    heatmap = cam.numpy()                       # (7,7)

    # 5) upsample & colorize
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 6) overlay
    overlay = cv2.addWeighted(heatmap, alpha, proc.astype("uint8"), 1-alpha, 0)
    return overlay

# Helpers to read uploads
def read_multipart(fs):
    arr = np.frombuffer(fs.read(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def read_base64(b64str):
    if "," in b64str:
        b64str = b64str.split(",",1)[1]
    arr = np.frombuffer(base64.b64decode(b64str), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # 1) load image
    img_bgr = None
    if "image" in request.files:
        img_bgr = read_multipart(request.files["image"])
    elif request.is_json and "image" in request.json:
        img_bgr = read_base64(request.json["image"])
    if img_bgr is None:
        return jsonify({"error":"No image provided"}), 400

    # 2) preprocess & score
    proc = preprocess(img_bgr)
    batch = tf.convert_to_tensor(proc[None,...])
    preds = model(batch, training=False)
    score = float(preds[0][0])
    label = "Real" if score>0.5 else "AI"

    # 3) gradcam
    overlay = make_gradcam_overlay(img_bgr)
    _,buf = cv2.imencode(".png", cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    hm = base64.b64encode(buf).decode()

    # 4) respond
    print(f"[Prediction] score={score:.4f}, label={label}", flush=True)
    return jsonify({
        "ai_generated_score": score,
        "label": label,
        "heatmap": hm
    })

if __name__=="__main__":
    app.run(debug=True)
