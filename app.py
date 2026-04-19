from flask import Flask, request, send_file
import cv2
import numpy as np
import io
from pdf2image import convert_from_bytes
from PIL import Image

app = Flask(__name__)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def crop_document(img):
    orig = img.copy()
    h, w = img.shape[:2]
    ratio = h / 1000.0
    small = cv2.resize(img, (int(w / ratio), 1000))

    # Converti in grigio
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Sfoca per ridurre rumore
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Soglia per trovare il documento bianco su sfondo scuro
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfologia per riempire buchi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Trova contorni
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return orig

    # Prende il contorno più grande (il documento)
    c = max(cnts, key=cv2.contourArea)

    # Verifica che sia abbastanza grande (almeno 20% dell'immagine)
    if cv2.contourArea(c) < (small.shape[0] * small.shape[1] * 0.2):
        return orig

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        # Raddrizza prospettiva
        pts = approx.reshape(4, 2).astype("float32") * ratio
        result = four_point_transform(orig, pts)
    else:
        # Bounding rect con piccolo padding
        x, y, bw, bh = cv2.boundingRect(c)
        pad = 10
        x = max(0, int(x * ratio) - pad)
        y = max(0, int(y * ratio) - pad)
        bw = min(w - x, int(bw * ratio) + pad * 2)
        bh = min(h - y, int(bh * ratio) + pad * 2)
        result = orig[y:y+bh, x:x+bw]

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
