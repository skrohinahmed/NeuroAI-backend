import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from werkzeug.utils import secure_filename

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# =========================================
# CONFIG
# =========================================
MODEL_PATH = "model/best_resnet18.pth"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

IMG_SIZE = 64
UNKNOWN_THRESHOLD = 0.55
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
# =========================================
# MODEL
# =========================================
class MyResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# =========================================
# LOAD CHECKPOINT
# =========================================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint["classes"]

model = MyResNet18(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================================
# GRAD-CAM
# =========================================
target_layers = [model.model.layer4[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# =========================================
# TRANSFORM
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================================
# HELPERS
# =========================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_heatmap_legend():
    drawing = Drawing(450, 80)

    gradient_colors = [
        colors.black,
        colors.HexColor("#2b0000"),
        colors.darkred,
        colors.red,
        colors.orange,
        colors.yellow,
        colors.white
    ]

    x = 20
    w = 55

    for c in gradient_colors:
        rect = Rect(x, 35, w, 25, fillColor=c, strokeColor=c)
        drawing.add(rect)
        x += w

    drawing.add(String(20, 15, "Low Attention", fontSize=10))
    drawing.add(String(170, 15, "Medium Attention", fontSize=10))
    drawing.add(String(330, 15, "High Attention", fontSize=10))
    drawing.add(String(20, 65, "Grad-CAM Heatmap Scale", fontSize=12))

    return drawing

def softmax_numpy(logits_tensor):
    probs = torch.softmax(logits_tensor, dim=1)
    return probs.detach().cpu().numpy()

def preprocess_pil(pil_img):
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def localize_image(image_path, top_k=3):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {
            "status": "error",
            "message": "Could not read image"
        }

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = preprocess_pil(pil)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = float(conf.item())
    pred_idx = int(pred.item())
    predicted_class = classes[pred_idx]

    probs_np = probs.detach().cpu().numpy()[0]
    top_indices = np.argsort(probs_np)[::-1][:top_k]
    top_predictions = [
        {
            "class": classes[int(i)],
            "confidence": float(probs_np[int(i)])
        }
        for i in top_indices
    ]

    if confidence < UNKNOWN_THRESHOLD:
        return {
            "status": "unknown",
            "message": "Unknown medical image",
            "confidence": confidence,
            "top_predictions": top_predictions
        }

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (rgb.shape[1], rgb.shape[0]))

    heatmap = (grayscale_cam * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    mask = grayscale_cam > 0.35
    darkened = (rgb * 0.85).astype(np.uint8)
    visualization = darkened.copy()

    visualization[mask] = (
        0.70 * rgb[mask] + 0.30 * colored_heatmap[mask]
    ).astype(np.uint8)

    output_filename = f"gradcam_{uuid.uuid4().hex}_{os.path.basename(image_path)}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return {
        "status": "success",
        "class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "localized_image": output_path,
        "original_image": image_path
    }

def generate_pdf(patient_name, patient_id, doctor_name, results):
    filename = f"report_{patient_id}_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = os.path.join(OUTPUT_FOLDER, filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Medical Grad-CAM Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    patient_data = [
        ["Patient Name", patient_name or "N/A"],
        ["Patient ID", patient_id or "N/A"],
        ["Doctor", doctor_name or "N/A"],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Model Format", "PyTorch (.pth)"],
        ["Threshold", str(UNKNOWN_THRESHOLD)]
    ]

    table = Table(patient_data)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 25))

    for i, r in enumerate(results):
        elements.append(Paragraph(f"Image {i+1}", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        result_table = Table([
            ["Predicted Class", r.get("class", "N/A")],
            ["Confidence", f"{r.get('confidence', 0):.4f}"]
        ])
        result_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(result_table)
        elements.append(Spacer(1, 10))

        for pred in r.get("top_predictions", []):
            elements.append(Paragraph(
                f"{pred['class']} : {pred['confidence']:.4f}",
                styles["BodyText"]
            ))

        elements.append(Spacer(1, 10))

        if os.path.exists(r["localized_image"]):
            elements.append(PDFImage(r["localized_image"], width=400, height=300))
            elements.append(Spacer(1, 15))

        elements.append(Paragraph("Heatmap Attention Scale", styles["Heading3"]))
        elements.append(create_heatmap_legend())
        elements.append(Spacer(1, 20))

        if i != len(results) - 1:
            elements.append(PageBreak())

    elements.append(Paragraph(
        "Grad-CAM highlights the most influential image regions used by the model for prediction.",
        styles["BodyText"]
    ))

    doc.build(elements)
    return pdf_path, filename

# =========================================
# ROUTES
# =========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "device": str(DEVICE),
        "total_classes": len(classes)
    })

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({
        "classes": classes,
        "total_classes": len(classes),
        "img_size": IMG_SIZE,
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "target_layer": "model.layer4[-1]"
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    patient_name = request.form.get("patient_name")
    patient_id = request.form.get("patient_id", "unknown")
    doctor_name = request.form.get("doctor_name")

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    success = []

    for f in files:
        if not f or not allowed_file(f.filename):
            results.append({
                "status": "error",
                "message": f"Invalid file: {getattr(f, 'filename', 'unknown')}"
            })
            continue

        filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        r = localize_image(path, top_k=3)
        results.append(r)

        if r["status"] == "success":
            success.append(r)

    pdf_url = None
    if success:
        _, pdf_file = generate_pdf(patient_name, patient_id, doctor_name, success)
        pdf_url = f"/download-report/{pdf_file}"

    return jsonify({
        "total": len(files),
        "success": len(success),
        "failed": len(files) - len(success),
        "pdf_url": pdf_url,
        "results": results
    })

@app.route("/download-report/<filename>", methods=["GET"])
def download_report(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0")