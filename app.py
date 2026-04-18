from flask import Flask, request, send_file
import cv2
import numpy as np
import io
from pdf2image import convert_from_bytes
from PIL import Image

app = Flask(__name__)

def crop_document(img):
    orig = img.copy()
    h, w = img.shape[:2]
    ratio = h / 1000.0
    small = cv2.resize(img, (int(w / ratio), 1000))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Trova area non bianca
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return orig
    
    # Prende il contorno più grande
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        # Raddrizza prospettiva
        pts = approx.reshape(4, 2) * ratio
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype="float32")
        (tl, tr, br, bl) = rect
        w2 = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        h2 = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        dst = np.array([[0,0],[w2-1,0],[w2-1,h2-1],[0,h2-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        result = cv2.warpPerspective(orig, M, (w2, h2))
    else:
        # Ritaglia bounding box con padding minimo
        x, y, bw, bh = cv2.boundingRect(c)
        x = int(x * ratio)
        y = int(y * ratio)
        bw = int(bw * ratio)
        bh = int(bh * ratio)
        pad = 15
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + pad * 2)
        bh = min(h - y, bh + pad * 2)
        result = orig[y:y+bh, x:x+bw]
    
    return result

def scan_document(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cropped = crop_document(img)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    final = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 10)
    pil_img = Image.fromarray(final)
    pdf_buf = io.BytesIO()
    pil_img.convert('RGB').save(pdf_buf, format='PDF', resolution=300)
    pdf_buf.seek(0)
    return pdf_buf.read()

@app.route('/scan', methods=['POST'])
def scan():
    data = request.data
    try:
        images = convert_from_bytes(data, dpi=300)
        img_pil = images[0]
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        image_bytes = buf.getvalue()
    except Exception:
        image_bytes = data
    scanned_pdf = scan_document(image_bytes)
    return send_file(io.BytesIO(scanned_pdf), mimetype='application/pdf')

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
