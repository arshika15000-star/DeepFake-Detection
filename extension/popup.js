const DEFAULT_BACKEND = 'http://127.0.0.1:8005';

let backendUrl = DEFAULT_BACKEND;
let currentModality = 'image';
let pollInterval = null;

// Load saved settings
chrome.storage.local.get(['backendUrl'], (data) => {
  backendUrl = data.backendUrl || DEFAULT_BACKEND;
  document.getElementById('backend-url-setting').value = backendUrl;
  checkBackendStatus();
});

// --- TABS ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// --- MODALITY PILLS ---
document.querySelectorAll('.pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    currentModality = pill.dataset.modality;
  });
});

// --- BACKEND STATUS ---
async function checkBackendStatus() {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  try {
    const res = await fetch(`${backendUrl}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok || res.status === 206) {
      dot.className = 'status-dot online';
      txt.textContent = 'Backend online ✓';
      txt.style.color = '#7ec8a0';
    } else {
      throw new Error('Bad status');
    }
  } catch {
    dot.className = 'status-dot offline';
    txt.textContent = 'Backend offline — start python app.py';
    txt.style.color = '#fb7185';
  }
}

// --- USE PAGE URL ---
document.getElementById('use-page-url-btn').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      document.getElementById('media-url').value = tabs[0].url;
      // Auto-detect from URL
      const url = tabs[0].url.toLowerCase();
      if (url.match(/\.(jpg|jpeg|png|webp|bmp|gif)(\?|$)/)) setModality('image');
      else if (url.match(/\.(mp4|avi|mov|mkv|webm)(\?|$)/)) setModality('video');
      else if (url.match(/\.(wav|mp3|m4a|flac)(\?|$)/)) setModality('audio');
    }
  });
});

function setModality(m) {
  currentModality = m;
  document.querySelectorAll('.pill').forEach(p => {
    p.classList.toggle('active', p.dataset.modality === m);
  });
}

// --- ANALYZE URL ---
document.getElementById('analyze-url-btn').addEventListener('click', analyzeUrl);
document.getElementById('media-url').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') analyzeUrl();
});

async function analyzeUrl() {
  const url = document.getElementById('media-url').value.trim();
  if (!url) { showError('Please enter a URL.'); return; }
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    showError('URL must start with http:// or https://');
    return;
  }

  showLoading('Sending to AI model…', 5);
  hideResult();
  hideError();

  try {
    const form = new FormData();
    form.append('url', url);
    form.append('modality', currentModality);

    const res = await fetch(`${backendUrl}/predict_url`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }
    const data = await res.json();
    if (data.job_id) {
      pollJob(data.job_id);
    } else {
      showResult(data);
    }
  } catch (err) {
    hideLoading();
    showError('Analysis failed: ' + err.message);
  }
}

function pollJob(jobId) {
  let attempt = 0;
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${backendUrl}/job/${jobId}`);
      const job = await res.json();
      const pct = Math.max(5, Math.min(95, job.progress || 0));
      showLoading(humanizeStatus(job.status), pct);

      if (job.status === 'completed') {
        clearInterval(pollInterval);
        hideLoading();
        showResult(job.result);
      } else if (job.status === 'failed') {
        clearInterval(pollInterval);
        hideLoading();
        showError('Analysis failed: ' + (job.error || 'Unknown error'));
      }
    } catch {
      attempt++;
      if (attempt > 15) { clearInterval(pollInterval); hideLoading(); showError('Connection lost.'); }
    }
  }, 1200);
}

function humanizeStatus(s) {
  const map = {
    'uploading': 'Uploading…', 'downloading_url': 'Downloading from URL…',
    'extracting_frames': 'Extracting frames…', 'analyzing_biometrics': 'Analyzing biometrics…',
    'running_neural_networks': 'Running neural networks…', 'generating_xai_artifacts': 'Generating XAI artifacts…',
    'tokenizing_text': 'Tokenizing text…', 'routing_to_model': 'Routing to model…',
  };
  return map[s] || (s ? s.replace(/_/g, ' ') : 'Analyzing…');
}

function showLoading(text, pct) {
  document.getElementById('loading-box').classList.add('show');
  document.getElementById('loading-text').textContent = text;
  document.getElementById('progress-fill').style.width = pct + '%';
}
function hideLoading() { document.getElementById('loading-box').classList.remove('show'); }

function showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = '⚠️ ' + msg;
  box.classList.add('show');
}
function hideError() { document.getElementById('error-box').classList.remove('show'); }

function showResult(result) {
  const verdict = result.prediction || 'UNKNOWN';
  const isFake = verdict === 'FAKE';
  const conf = ((result.confidence || 0) * 100).toFixed(1);

  document.getElementById('result-verdict').textContent = isFake ? '🚨 DEEPFAKE' : '✅ AUTHENTIC';
  document.getElementById('result-verdict').className = 'result-verdict ' + (isFake ? 'fake' : 'real');
  document.getElementById('result-fill').style.width = conf + '%';
  document.getElementById('result-fill').className = 'result-fill ' + (isFake ? 'fake' : 'real');
  document.getElementById('result-meta').textContent = `Confidence: ${conf}%`;

  // Findings
  const findingsEl = document.getElementById('result-findings');
  const findings = result.forensics?.findings || [];
  if (findings.length > 0) {
    findingsEl.innerHTML = findings.slice(0, 5).map(f => `<div class="finding-item">${f}</div>`).join('');
  } else {
    findingsEl.innerHTML = '';
  }

  document.getElementById('result-box').classList.add('show');
}
function hideResult() { document.getElementById('result-box').classList.remove('show'); }

document.getElementById('reset-btn').addEventListener('click', () => {
  hideResult();
  hideError();
  document.getElementById('media-url').value = '';
});

// --- PAGE IMAGES TAB ---
document.getElementById('scan-page-btn').addEventListener('click', () => {
  const container = document.getElementById('page-images');
  container.innerHTML = '<div class="empty-state">Scanning page…</div>';

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      func: () => {
        const imgs = Array.from(document.images)
          .filter(img => img.naturalWidth > 100 && img.naturalHeight > 100)
          .slice(0, 20)
          .map(img => ({ src: img.src, width: img.naturalWidth, height: img.naturalHeight }));
        return imgs;
      }
    }, (results) => {
      const imgs = results?.[0]?.result || [];
      if (imgs.length === 0) {
        container.innerHTML = '<div class="empty-state">No large images found on this page.</div>';
        return;
      }
      container.innerHTML = imgs.map((img, i) =>
        `<div class="page-img-item" data-src="${img.src}" data-idx="${i}">
          <img class="page-img-thumb" src="${img.src}" onerror="this.style.background='#333'" />
          <span class="page-img-url">${img.src}</span>
          <button class="page-img-btn" data-src="${img.src}">Analyze</button>
        </div>`
      ).join('');

      container.querySelectorAll('.page-img-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const src = btn.dataset.src;
          document.getElementById('media-url').value = src;
          setModality('image');
          // Switch to URL tab
          document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
          document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
          document.querySelector('[data-tab="url-tab"]').classList.add('active');
          document.getElementById('url-tab').classList.add('active');
          analyzeUrl();
        });
      });
    });
  });
});

// --- SETTINGS ---
document.getElementById('save-settings-btn').addEventListener('click', () => {
  const val = document.getElementById('backend-url-setting').value.trim() || DEFAULT_BACKEND;
  backendUrl = val;
  chrome.storage.local.set({ backendUrl: val }, () => {
    const btn = document.getElementById('save-settings-btn');
    btn.textContent = '✅ Saved!';
    setTimeout(() => { btn.textContent = '💾 Save Settings'; }, 1500);
    checkBackendStatus();
  });
});
