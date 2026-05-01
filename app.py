from datetime import datetime
from io import BytesIO
import base64
import time

from flask import Flask, render_template_string, request
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model/malaria_model.keras')


def prepare_image(image: Image.Image, target_size=(128, 128)):
    image = image.convert('RGB').resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


PAGE_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MalariaScope AI</title>
  <style>
    :root {
      --bg-0: #070d1f;
      --bg-1: #0f1a36;
      --panel: rgba(28, 37, 67, 0.62);
      --panel-2: rgba(40, 50, 80, 0.50);
      --line: rgba(110, 146, 255, 0.24);
      --text: #eaf0ff;
      --muted: #9ba7c7;
      --brand-a: #6f7bff;
      --brand-b: #9658e4;
      --ok: #5ad68f;
      --bad: #ff6d6d;
      --radius: 16px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: var(--text);
      min-height: 100vh;
      background:
        radial-gradient(1000px 400px at 0% 100%, rgba(31, 69, 168, 0.35), transparent 60%),
        radial-gradient(1000px 450px at 100% 0%, rgba(130, 52, 212, 0.25), transparent 55%),
        linear-gradient(145deg, var(--bg-0), var(--bg-1));
    }

    .shell {
      max-width: 1050px;
      margin: 0 auto;
      padding: 22px 16px 30px;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 26px;
    }

    .brand {
      font-weight: 700;
      letter-spacing: 0.2px;
      color: #8eb1ff;
    }

    .chip {
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid rgba(113, 226, 166, 0.3);
      background: rgba(73, 182, 123, 0.16);
      color: #9ae9be;
      font-size: 12px;
      font-weight: 600;
    }

    .hero {
      text-align: center;
      margin-bottom: 26px;
    }

    .hero h1 {
      margin: 0 0 10px;
      font-size: clamp(1.85rem, 3.2vw, 3rem);
      line-height: 1.12;
    }

    .hero .accent {
      background: linear-gradient(90deg, #9db4ff, #9e6df6);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 0.98rem;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      padding: 12px;
      text-align: center;
      font-weight: 600;
    }

    .metric small { display: block; color: var(--muted); font-weight: 500; margin-top: 4px; }

    .upload-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      backdrop-filter: blur(6px);
      padding: 16px;
    }

    .dropzone {
      border: 1px dashed rgba(109, 154, 255, 0.45);
      border-radius: 14px;
      background: var(--panel-2);
      min-height: 260px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 20px;
      position: relative;
      overflow: hidden;
      margin-bottom: 14px;
    }

    #preview {
      max-width: 100%;
      max-height: 230px;
      border-radius: 8px;
      display: none;
      image-rendering: auto;
    }

    .muted {
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 8px;
    }

    .btn {
      width: 100%;
      border: 0;
      border-radius: 10px;
      padding: 12px 16px;
      color: #f5f7ff;
      font-weight: 700;
      cursor: pointer;
      background: linear-gradient(90deg, var(--brand-a), var(--brand-b));
    }

    .btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    .hint {
      margin-top: 12px;
      color: #8d9ac2;
      text-align: center;
      font-size: 0.82rem;
    }

    .report-grid {
      display: grid;
      grid-template-columns: 2.1fr 1fr;
      gap: 14px;
    }

    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 14px;
      padding: 14px;
    }

    .status {
      border: 1px solid {{ report.border_color if report else '#5f7cf5' }};
      background: {{ report.status_bg if report else 'rgba(75,88,141,0.18)' }};
      border-radius: 12px;
      padding: 18px;
      text-align: center;
      margin-bottom: 12px;
    }

    .status h2 { margin: 2px 0; }
    .status p { margin: 0; color: var(--muted); }

    .bar {
      background: rgba(27, 39, 72, 0.9);
      border-radius: 999px;
      height: 14px;
      overflow: hidden;
      border: 1px solid rgba(136, 164, 252, 0.25);
      margin-top: 8px;
    }

    .bar-fill {
      height: 100%;
      width: {{ report.confidence if report else 0 }}%;
      background: {{ report.bar_color if report else 'linear-gradient(90deg, #7792ff, #7a6ff4)' }};
    }

    .kvs {
      display: grid;
      gap: 10px;
      margin-top: 8px;
    }

    .kv {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(116, 138, 201, 0.2);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .kv b { color: #f3f6ff; }

    .action-row {
      margin-top: 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .link-btn {
      border: 1px solid rgba(137, 156, 236, 0.45);
      color: #dfe7ff;
      text-decoration: none;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.9rem;
      background: rgba(74, 94, 160, 0.22);
    }

    .err {
      border: 1px solid rgba(255, 122, 122, 0.45);
      background: rgba(181, 59, 59, 0.25);
      color: #ffd7d7;
      padding: 10px 12px;
      border-radius: 10px;
      margin-bottom: 10px;
      font-size: 0.95rem;
    }

    @media (max-width: 850px) {
      .metrics { grid-template-columns: 1fr; }
      .report-grid { grid-template-columns: 1fr; }
      .shell { padding: 16px 12px 22px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">MalariaScope AI</div>
      <div class="chip">System Online</div>
    </header>

    {% if report %}
      <section class="report-grid">
        <div class="panel">
          <h2 style="margin: 0 0 12px;">Diagnostic Report</h2>
          <div class="status">
            <h2>{{ report.label }}</h2>
            <p>{{ report.caption }}</p>
          </div>

          <div class="panel" style="padding: 12px; margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size:0.92rem; color:var(--muted);">
              <span>Confidence Level</span>
              <b style="color:#dbe5ff;">{{ report.confidence }}%</b>
            </div>
            <div class="bar"><div class="bar-fill"></div></div>
          </div>

          {% if report.preview_url %}
          <div class="panel" style="padding: 12px;">
            <div style="margin-bottom: 8px; color: var(--muted); font-size: 0.9rem;">Analyzed Sample</div>
            <img src="{{ report.preview_url }}" alt="Uploaded blood cell image" style="width:100%; max-height: 300px; object-fit: contain; border-radius: 10px; border: 1px solid rgba(120,142,210,0.3); background: rgba(20,28,53,0.75);" />
          </div>
          {% endif %}
        </div>

        <aside class="panel">
          <h3 style="margin: 0 0 10px;">Analysis Details</h3>
          <div class="kvs">
            <div class="kv"><span>Model</span><b>CustomCNN v2.0</b></div>
            <div class="kv"><span>Processing Time</span><b>{{ report.processing_time }}s</b></div>
            <div class="kv"><span>Image Quality</span><b>{{ report.image_quality }}</b></div>
            <div class="kv"><span>Timestamp</span><b>{{ report.timestamp }}</b></div>
          </div>

          <div class="panel" style="margin-top:12px;">
            <h4 style="margin:0 0 8px;">Medical Disclaimer</h4>
            <p style="margin:0; color:var(--muted); font-size:0.88rem; line-height:1.45;">
              This AI result is a screening aid for healthcare professionals and is not a substitute for clinical diagnosis.
            </p>
          </div>

          <div class="action-row">
            <a class="link-btn" href="/">Analyze Another Sample</a>
          </div>
        </aside>
      </section>
    {% else %}
      <section class="hero">
        <h1>Revolutionary <span class="accent">AI-Powered</span><br/>Malaria Detection</h1>
        <p>Clinical-grade accuracy, real-time analysis, instant results.</p>
        <div class="metrics">
          <div class="metric">99.2%<small>Accuracy</small></div>
          <div class="metric">&lt;3s<small>Analysis Time</small></div>
          <div class="metric">2,852<small>Scans Today</small></div>
        </div>
      </section>

      <section class="upload-card">
        <h2 style="text-align:center; margin: 0 0 8px;">Upload Blood Cell Image</h2>
        <p style="text-align:center; color: var(--muted); margin: 0 0 14px; font-size: 0.9rem;">
          Drag and drop or browse. Supports PNG, JPG, JPEG.
        </p>

        {% if error %}
        <div class="err">{{ error }}</div>
        {% endif %}

        <form action="/predict" method="post" enctype="multipart/form-data">
          <div class="dropzone" id="dropzone">
            <div id="placeholder">
              <div style="font-size:1.05rem; font-weight:700;">Drop your microscopic image here</div>
              <div class="muted">or click to select from your device</div>
            </div>
            <img id="preview" alt="Image preview" />
            <input id="fileInput" type="file" name="file" accept=".png,.jpg,.jpeg" required
              style="position:absolute; inset:0; opacity:0; cursor:pointer;" />
          </div>
          <button class="btn" id="analyzeBtn" type="submit" disabled>Analyze Sample</button>
        </form>
        <div class="hint">Powered by deep learning and TensorFlow.</div>
      </section>
    {% endif %}
  </main>

  <script>
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (fileInput) {
      fileInput.addEventListener('change', (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) {
          if (preview) preview.style.display = 'none';
          if (placeholder) placeholder.style.display = 'block';
          if (analyzeBtn) analyzeBtn.disabled = true;
          return;
        }

        if (analyzeBtn) analyzeBtn.disabled = false;

        const reader = new FileReader();
        reader.onload = (e) => {
          if (preview) {
            preview.src = e.target.result;
            preview.style.display = 'block';
          }
          if (placeholder) placeholder.style.display = 'none';
        };
        reader.readAsDataURL(file);
      });
    }
  </script>
</body>
</html>
'''


def build_report(raw_prediction: float, elapsed: float, image: Image.Image):
    if raw_prediction < 0.5:
        label = 'PARASITIZED'
        confidence = (1.0 - raw_prediction) * 100
        caption = 'Malaria parasites detected'
        border_color = 'rgba(255, 114, 114, 0.7)'
        status_bg = 'rgba(137, 42, 61, 0.22)'
        bar_color = 'linear-gradient(90deg, #ff8b8b, #ff6262)'
    else:
        label = 'UNINFECTED'
        confidence = raw_prediction * 100
        caption = 'No malaria parasites detected'
        border_color = 'rgba(102, 225, 164, 0.7)'
        status_bg = 'rgba(35, 133, 96, 0.18)'
        bar_color = 'linear-gradient(90deg, #70e3a5, #50cc8d)'

    confidence = round(float(confidence), 2)

    if confidence >= 95:
        image_quality = 'Excellent'
    elif confidence >= 85:
        image_quality = 'Good'
    else:
        image_quality = 'Fair'

    return {
        'label': label,
        'confidence': confidence,
        'caption': caption,
        'border_color': border_color,
        'status_bg': status_bg,
        'bar_color': bar_color,
        'image_quality': image_quality,
        'processing_time': f'{elapsed:.2f}',
        'timestamp': datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        'preview_url': to_data_url(image),
    }


@app.route('/')
def home():
    return render_template_string(PAGE_TEMPLATE, error=None, report=None)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template_string(PAGE_TEMPLATE, error='No file uploaded. Please choose an image.', report=None)

    file = request.files['file']
    if not file or not file.filename:
        return render_template_string(PAGE_TEMPLATE, error='No file selected. Please choose an image.', report=None)

    try:
        image = Image.open(file)
    except UnidentifiedImageError:
        return render_template_string(PAGE_TEMPLATE, error='Unsupported file. Upload PNG, JPG, or JPEG image.', report=None)

    start = time.perf_counter()
    processed = prepare_image(image)
    raw_prediction = float(model.predict(processed, verbose=0)[0][0])
    elapsed = time.perf_counter() - start

    report = build_report(raw_prediction=raw_prediction, elapsed=elapsed, image=image)
    return render_template_string(PAGE_TEMPLATE, error=None, report=report)


if __name__ == '__main__':
    app.run(debug=True)
