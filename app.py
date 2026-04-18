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
    
    # Ridimensiona per elaborazione
    ratio = h / 800.0
    small = cv2.resize(img, (int(w / ratio), 800))
    
    # Converti in grigio e trova bordi
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    
    # Dilata i bordi per connettere linee spezzate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=2)
    
    # Trova contorni
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break
    
    if doc_cnt is not None:
        # Raddrizza prospettiva
        pts = doc_cnt.reshape(4, 2) * ratio
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
        # Nessun documento trovato: ritaglia bordi bianchi automaticamente
        gray2 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray2, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w3, h3 = cv2.boundingRect(coords)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w3 = min(orig.shape[1] - x, w3 + padding * 2)
            h3 = min(orig.shape[0] - y, h3 + padding * 2)
            result = orig[y:y+h3, x:x+w3]
        else:
            result = orig
    
    return result

def scan_document(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Ritaglia e raddrizza
    cropped = crop_document(img)
    
    # Converti in bianco/nero stile scanner
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    final = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 10)
    
    # Converti in PDF
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
