from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

def scan_document(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = img.copy()
    ratio = img.shape[0] / 500.0
    small = cv2.resize(img, (int(img.shape[1] / ratio), 500))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    doc = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break
    if doc is not None:
        pts = doc.reshape(4, 2) * ratio
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                         pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")
        (tl, tr, br, bl) = rect
        w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (w, h))
        result = warped
    else:
        result = orig
    gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    final = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 10)
    _, buf = cv2.imencode('.png', final)
    return buf.tobytes()

@app.route('/scan', methods=['POST'])
def scan():
    image_bytes = request.data
    scanned = scan_document(image_bytes)
    return send_file(io.BytesIO(scanned), mimetype='image/png')

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
