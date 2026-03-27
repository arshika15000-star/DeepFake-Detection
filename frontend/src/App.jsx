import React, { useState } from 'react';
import axios from 'axios';
import { Play, UploadCloud, Camera, Mic, Type, Link2, ArrowRight, X, Mail, User, Lock, MessageSquare, Phone, Send } from 'lucide-react';
import ScannerView from './components/ScannerView';
import CaptureView from './components/CaptureView';
import ResultDashboard from './components/ResultDashboard';

const getApiBaseUrl = () => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  return "http://127.0.0.1:8000";
};

const API_BASE = getApiBaseUrl();

// --- STARRY BACKGROUND ---
function StarBackground() {
  const stars = Array.from({ length: 70 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    top: Math.random() * 100,
    size: Math.random() * 3 + 1,
    delay: Math.random() * 5,
  }));
  return (
    <div className="star-bg">
      {stars.map((s) => (
        <div key={s.id} className="star" style={{ left: `${s.left}%`, top: `${s.top}%`, width: s.size, height: s.size, animationDelay: `${s.delay}s` }} />
      ))}
    </div>
  );
}

// --- LOGIN / SIGNUP MODAL ---
function AuthModal({ onClose }) {
  const [tab, setTab] = useState('login');
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [msg, setMsg] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (tab === 'signup' && form.password !== form.confirm) {
      setMsg('❌ Passwords do not match.');
      return;
    }
    // Simulate local auth (no server — just demo)
    const key = `df_user_${form.email}`;
    if (tab === 'signup') {
      if (localStorage.getItem(key)) { setMsg('❌ Email already registered.'); return; }
      localStorage.setItem(key, JSON.stringify({ name: form.name, email: form.email, password: form.password }));
      setMsg('✅ Account created! You can now log in.');
      setTab('login');
    } else {
      const stored = localStorage.getItem(key);
      if (!stored) { setMsg('❌ Account not found. Please sign up first.'); return; }
      const user = JSON.parse(stored);
      if (user.password !== form.password) { setMsg('❌ Incorrect password.'); return; }
      localStorage.setItem('df_current_user', JSON.stringify(user));
      setMsg(`✅ Welcome back, ${user.name || user.email}!`);
      setTimeout(onClose, 1200);
    }
  };

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-md rounded-3xl p-8 shadow-2xl" style={{ background: 'rgba(15,15,25,0.97)', border: '1px solid rgba(126,200,160,0.3)' }}>
        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 transition-colors" style={{ color: '#7ec8a0' }}><X size={18} /></button>
        <h2 className="text-2xl font-bold mb-1 text-center" style={{ color: '#7ec8a0' }}>
          {tab === 'login' ? 'Welcome Back' : 'Create Account'}
        </h2>
        <p className="text-center text-sm text-gray-400 mb-6">DeepFake Detection Platform</p>

        {/* Tabs */}
        <div className="flex rounded-xl overflow-hidden border mb-6" style={{ borderColor: 'rgba(126,200,160,0.2)' }}>
          {['login', 'signup'].map(t => (
            <button key={t} onClick={() => { setTab(t); setMsg(''); }} className="flex-1 py-2 text-sm font-bold transition-all"
              style={{ background: tab === t ? 'rgba(126,200,160,0.2)' : 'transparent', color: tab === t ? '#7ec8a0' : '#888' }}>
              {t === 'login' ? 'Log In' : 'Sign Up'}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {tab === 'signup' && (
            <div className="relative">
              <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input required placeholder="Full Name" value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors"
                style={{ borderColor: 'rgba(126,200,160,0.2)' }} />
            </div>
          )}
          <div className="relative">
            <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input required type="email" placeholder="Email Address" value={form.email} onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
              className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors"
              style={{ borderColor: 'rgba(126,200,160,0.2)' }} />
          </div>
          <div className="relative">
            <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input required type="password" placeholder="Password" value={form.password} onChange={e => setForm(p => ({ ...p, password: e.target.value }))}
              className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors"
              style={{ borderColor: 'rgba(126,200,160,0.2)' }} />
          </div>
          {tab === 'signup' && (
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input required type="password" placeholder="Confirm Password" value={form.confirm} onChange={e => setForm(p => ({ ...p, confirm: e.target.value }))}
                className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors"
                style={{ borderColor: 'rgba(126,200,160,0.2)' }} />
            </div>
          )}
          {msg && <p className="text-xs text-center font-medium py-2 px-3 rounded-lg" style={{ background: msg.startsWith('✅') ? 'rgba(126,200,160,0.1)' : 'rgba(251,113,133,0.1)', color: msg.startsWith('✅') ? '#7ec8a0' : '#fb7185' }}>{msg}</p>}
          <button type="submit" id={tab === 'login' ? 'login-submit-btn' : 'signup-submit-btn'}
            className="w-full py-3 rounded-xl font-bold text-black transition-all hover:scale-105 active:scale-95 mt-2"
            style={{ background: 'linear-gradient(135deg, #7ec8a0, #5aaa80)' }}>
            {tab === 'login' ? 'Log In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  );
}

// --- CONTACT US MODAL ---
function ContactModal({ onClose }) {
  const [form, setForm] = useState({ name: '', email: '', subject: '', message: '' });
  const [sent, setSent] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simulate sending
    setSent(true);
    setTimeout(onClose, 2500);
  };

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-lg rounded-3xl p-8 shadow-2xl" style={{ background: 'rgba(15,15,25,0.97)', border: '1px solid rgba(99,144,255,0.3)' }}>
        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 transition-colors" style={{ color: '#6390ff' }}><X size={18} /></button>
        <h2 className="text-2xl font-bold mb-1 text-center" style={{ color: '#6390ff' }}>Contact Us</h2>
        <p className="text-center text-sm text-gray-400 mb-6">We'd love to hear from you. Drop us a message!</p>

        {sent ? (
          <div className="text-center py-12">
            <div className="text-5xl mb-4">✅</div>
            <h3 className="text-xl font-bold text-[#6390ff] mb-2">Message Sent!</h3>
            <p className="text-gray-400 text-sm">Thank you for reaching out. We'll get back to you soon.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="relative">
                <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input required placeholder="Your Name" value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                  className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#6390ff] transition-colors"
                  style={{ borderColor: 'rgba(99,144,255,0.2)' }} />
              </div>
              <div className="relative">
                <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input required type="email" placeholder="Email" value={form.email} onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
                  className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#6390ff] transition-colors"
                  style={{ borderColor: 'rgba(99,144,255,0.2)' }} />
              </div>
            </div>
            <div className="relative">
              <MessageSquare size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input required placeholder="Subject" value={form.subject} onChange={e => setForm(p => ({ ...p, subject: e.target.value }))}
                className="w-full pl-9 pr-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#6390ff] transition-colors"
                style={{ borderColor: 'rgba(99,144,255,0.2)' }} />
            </div>
            <textarea required rows={5} placeholder="Your message..." value={form.message} onChange={e => setForm(p => ({ ...p, message: e.target.value }))}
              className="w-full px-4 py-3 rounded-xl bg-white/5 border text-white text-sm focus:outline-none focus:border-[#6390ff] transition-colors resize-none"
              style={{ borderColor: 'rgba(99,144,255,0.2)' }} />
            <button type="submit" id="contact-submit-btn"
              className="w-full py-3 rounded-xl font-bold text-white flex items-center justify-center gap-2 transition-all hover:scale-105 active:scale-95"
              style={{ background: 'linear-gradient(135deg, #6390ff, #4a72d1)' }}>
              <Send size={16} /> Send Message
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState('home');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isCaptureViewOpen, setIsCaptureViewOpen] = useState(false);
  const [selectedModality, setSelectedModality] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [urlInputs, setUrlInputs] = useState({ image: '', video: '', audio: '' });
  const [showAuth, setShowAuth] = useState(false);
  const [showContact, setShowContact] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, status: 'initializing' });

  const pollJobStatus = async (jobId) => {
    try {
      const res = await axios.get(`${API_BASE}/job/${jobId}`);
      const job = res.data;
      setProgress({ percent: job.progress, status: job.status });
      if (job.status === 'completed') {
        setAnalysisResult(job.result);
        setView('result');
      } else if (job.status === 'failed') {
        throw new Error(job.error || "Analysis failed during processing");
      } else {
        setTimeout(() => pollJobStatus(jobId), 1000);
      }
    } catch (err) {
      console.error("Polling error:", err);
      alert("Analysis Failed: " + (err.response?.data?.detail || err.message));
      setView('home');
    }
  };

  const startAnalysis = async (inputData, modality) => {
    setSelectedModality(modality);
    setView('scanning');
    setProgress({ percent: 0, status: 'uploading' });
    try {
      const formData = new FormData();
      let endpoint = '';
      if (modality === 'text') {
        endpoint = '/predict_text';
        formData.append('current_text', inputData);
      } else {
        endpoint = modality === 'image' ? '/predict_image' : modality === 'audio' ? '/predict_audio' : '/predict';
        formData.append('file', inputData);
      }
      const response = await axios.post(`${API_BASE}${endpoint}`, formData);
      if (response.data.job_id) {
        pollJobStatus(response.data.job_id);
      } else {
        setAnalysisResult(response.data);
        setView('result');
      }
    } catch (err) {
      console.error(err);
      alert("Analysis Upload Failed: " + (err.response?.data?.detail || err.message));
      setView('home');
    }
  };

  const handleUrlAnalysis = async (modality) => {
    const url = urlInputs[modality]?.trim();
    if (!url) { alert('Please paste a URL first.'); return; }
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      alert('URL must start with http:// or https://');
      return;
    }
    setSelectedModality(modality);
    setView('scanning');
    setProgress({ percent: 0, status: 'downloading_url' });
    try {
      const formData = new FormData();
      formData.append('url', url);
      formData.append('modality', modality);
      const response = await axios.post(`${API_BASE}/predict_url`, formData);
      if (response.data.job_id) {
        pollJobStatus(response.data.job_id);
      } else {
        setAnalysisResult(response.data);
        setView('result');
      }
    } catch (err) {
      console.error(err);
      alert('URL Analysis Failed: ' + (err.response?.data?.detail || err.message));
      setView('home');
    }
  };

  const ALLOWED_TYPES = {
    image: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
    video: ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'],
    audio: ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac', 'audio/webm'],
  };
  const MAX_FILE_MB = { image: 10, video: 200, audio: 50 };

  const handleFileUpload = (e, modality) => {
    const f = e.target.files[0];
    if (!f) return;
    const allowed = ALLOWED_TYPES[modality] || [];
    const maxMB = MAX_FILE_MB[modality] || 100;
    const sizeMB = f.size / (1024 * 1024);
    if (!allowed.includes(f.type)) {
      alert(`❌ Invalid file type for ${modality}.\nAllowed: ${allowed.map(t => t.split('/')[1].toUpperCase()).join(', ')}`);
      e.target.value = '';
      return;
    }
    if (sizeMB > maxMB) {
      alert(`❌ File too large (${sizeMB.toFixed(1)} MB).\nMaximum allowed for ${modality}: ${maxMB} MB.`);
      e.target.value = '';
      return;
    }
    startAnalysis(f, modality);
  };

  const handleDragOver = (e) => { e.preventDefault(); if (!isDragging) setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (!f) return;
    if (f.type.startsWith('image/')) handleFileUpload({ target: { files: [f] } }, 'image');
    else if (f.type.startsWith('video/')) handleFileUpload({ target: { files: [f] } }, 'video');
    else if (f.type.startsWith('audio/')) handleFileUpload({ target: { files: [f] } }, 'audio');
    else alert('❌ Unsupported file drop type.');
  };

  const modalityColors = {
    image: { accent: '#7ec8a0', glow: 'rgba(126,200,160,0.3)', bg: '#7ec8a0', btnBg: '#6ab38c', border: 'rgba(126,200,160,0.4)', inputBorder: 'rgba(126,200,160,0.3)', dark: false },
    video: { accent: '#6390ff', glow: 'rgba(99,144,255,0.3)', bg: '#6390ff', btnBg: '#4a72d1', border: 'rgba(99,144,255,0.4)', inputBorder: 'rgba(99,144,255,0.3)', dark: false },
    audio: { accent: '#ff9063', glow: 'rgba(255,144,99,0.3)', bg: '#ff9063', btnBg: '#e07548', border: 'rgba(255,144,99,0.4)', inputBorder: 'rgba(255,144,99,0.3)', dark: false },
    text:  { accent: '#c87eff', glow: 'rgba(200,126,255,0.3)', bg: '#c87eff', btnBg: '#a55cd9', border: 'rgba(200,126,255,0.4)', inputBorder: 'rgba(200,126,255,0.3)', dark: false },
  };

  const ModalityCard = ({ id, modality, icon: Icon, title, description, uploadAccept, uploadId, captureBtnText, captureIcon: CaptureIcon }) => {
    const c = modalityColors[modality];
    return (
      <div id={`${modality}-card`} className="glass-morphism rounded-3xl p-8 flex flex-col items-center text-center border transition-all hover:scale-105 shadow-xl relative overflow-hidden group"
        style={{ borderColor: c.border, boxShadow: `0 4px 30px ${c.glow}00` }}
        onMouseEnter={e => e.currentTarget.style.boxShadow = `0 4px 30px ${c.glow}`}
        onMouseLeave={e => e.currentTarget.style.boxShadow = `0 4px 30px ${c.glow}00`}>
        <div className="absolute inset-0 bg-gradient-to-b to-transparent opacity-0 group-hover:opacity-100 transition-opacity" style={{ backgroundImage: `linear-gradient(to bottom, ${c.accent}18, transparent)` }} />
        <Icon size={48} className="mb-4" style={{ color: c.accent }} />
        <h2 className="text-2xl font-bold mb-2">{title}</h2>
        <p className="text-sm text-gray-400 mb-6 flex-1">{description}</p>
        <div className="w-full space-y-3 z-10">
          {modality !== 'text' ? (
            <>
              <button onClick={() => document.getElementById(uploadId).click()}
                className="w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all hover:opacity-90"
                style={{ background: c.bg, color: modality === 'image' ? '#000' : '#fff' }}>
                <UploadCloud size={18} /> Upload {title}
              </button>
              <input id={uploadId} type="file" accept={uploadAccept} className="hidden" onChange={(e) => handleFileUpload(e, modality)} />
              <button onClick={() => { setSelectedModality(modality); setIsCaptureViewOpen(true); }}
                className="w-full py-3 rounded-xl font-bold border flex items-center justify-center gap-2 transition-all hover:opacity-90"
                style={{ borderColor: c.accent, color: c.accent, background: `${c.accent}12` }}>
                {CaptureIcon && <CaptureIcon size={18} />} {captureBtnText}
              </button>
              {/* URL input row */}
              <div className="flex gap-2">
                <input id={`${modality}-url-input`} type="url" placeholder={`Paste ${title.toLowerCase()} URL…`}
                  value={urlInputs[modality] || ''}
                  onChange={e => setUrlInputs(p => ({ ...p, [modality]: e.target.value }))}
                  onKeyDown={e => e.key === 'Enter' && handleUrlAnalysis(modality)}
                  className="flex-1 px-3 py-2 rounded-xl text-sm text-white placeholder-gray-500 focus:outline-none transition-colors"
                  style={{ background: 'rgba(0,0,0,0.4)', border: `1px solid ${c.inputBorder}` }} />
                <button onClick={() => handleUrlAnalysis(modality)} title="Analyze URL"
                  className="px-3 py-2 rounded-xl transition-all hover:scale-110"
                  style={{ background: `${c.accent}22`, border: `1px solid ${c.accent}66`, color: c.accent }}>
                  <ArrowRight size={16} />
                </button>
              </div>
            </>
          ) : (
            <>
              <button onClick={() => { const txt = prompt("Paste your text for semantic AI analysis:"); if (txt) startAnalysis(txt, 'text'); }}
                className="w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all hover:opacity-90"
                style={{ background: c.bg, color: '#fff' }}>
                <Type size={18} /> Paste Text
              </button>
              <button className="w-full py-3 rounded-xl font-bold border opacity-40 cursor-not-allowed flex items-center justify-center gap-2"
                style={{ borderColor: c.accent, color: c.accent }}>
                <UploadCloud size={18} /> Upload Document (Coming Soon)
              </button>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen relative overflow-hidden"
      style={{ backgroundColor: '#000', color: '#fff' }}
      onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
      <StarBackground />

      {/* Modals */}
      {showAuth && <AuthModal onClose={() => setShowAuth(false)} />}
      {showContact && <ContactModal onClose={() => setShowContact(false)} />}

      {isDragging && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm border-4 border-dashed border-[#7ec8a0]">
          <h2 className="text-4xl font-bold text-[#7ec8a0] animate-pulse">Drop File to Analyze</h2>
        </div>
      )}

      {/* NAVBAR */}
      <nav className="relative z-50 flex items-center justify-between px-10 py-6">
        <div className="flex items-center gap-4 cursor-pointer" onClick={() => setView('home')}>
          <div className="w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold text-black" style={{ backgroundColor: '#7ec8a0', boxShadow: '0 0 20px rgba(126, 200, 160, 0.6)' }}>DF</div>
          <span className="text-3xl font-bold tracking-wide" style={{ color: '#7ec8a0', textShadow: '0 0 10px rgba(126, 200, 160, 0.4)' }}>DeepFake</span>
        </div>
        <div className="hidden md:flex items-center gap-6">
          <button onClick={() => setView('home')} className="nav-link font-semibold text-sm tracking-wide text-white hover:text-[#7ec8a0] transition-colors">Home</button>
          <button className="nav-link font-semibold text-sm tracking-wide text-white hover:text-[#7ec8a0] transition-colors">About</button>
          <button id="login-nav-btn" onClick={() => setShowAuth(true)}
            className="px-5 py-2 rounded-full text-sm font-bold border transition-all hover:scale-105"
            style={{ borderColor: '#7ec8a0', color: '#7ec8a0', background: 'rgba(126,200,160,0.08)' }}>
            Login / Sign Up
          </button>
          <button id="contact-nav-btn" onClick={() => setShowContact(true)}
            className="px-5 py-2 rounded-full text-sm font-bold transition-all hover:scale-105"
            style={{ background: 'linear-gradient(135deg,#6390ff,#4a72d1)', color: '#fff' }}>
            Contact Us
          </button>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <main className="relative z-10 w-full min-h-[85vh] flex items-center justify-center px-10">
        {view === 'home' && (
          <div className="w-full max-w-[90vw] flex flex-col items-center mt-10 space-y-12">
            <div className="text-center space-y-4">
              <h1 className="text-5xl lg:text-7xl font-bold tracking-tight glow-text leading-tight">Deepfake Intelligence</h1>
              <p className="text-xl text-gray-300 max-w-2xl mx-auto">Select a modality to begin forensic analysis. Upload, capture live, or paste a URL.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8 w-full pb-20">
              <ModalityCard modality="image" icon={UploadCloud} title="Image" uploadAccept="image/*" uploadId="image-upload"
                captureBtnText="Live Snapshot" captureIcon={Camera}
                description="Detect AI-generated faces, GAN artifacts, and manipulated pixels in photos." />
              <ModalityCard modality="video" icon={Play} title="Video" uploadAccept="video/*" uploadId="video-upload"
                captureBtnText="Record Video" captureIcon={Camera}
                description="Analyze temporal coherence, rPPG pulse, and frame-by-frame deepfake signatures." />
              <ModalityCard modality="audio" icon={Mic} title="Audio" uploadAccept="audio/*" uploadId="audio-upload"
                captureBtnText="Record Audio" captureIcon={Mic}
                description="Extract acoustic fingerprints and detect cloned or synthetic AI voices." />
              <ModalityCard modality="text" icon={Type} title="Text" description="Evaluate linguistic syntax, perplexity, and AI generator writing patterns." />
            </div>
          </div>
        )}

        {view === 'scanning' && selectedModality && (
          <ScannerView modality={selectedModality} progress={progress} />
        )}

        {isCaptureViewOpen && (
          <CaptureView modality={selectedModality}
            onCapture={(file) => { setIsCaptureViewOpen(false); startAnalysis(file, selectedModality); }}
            onClose={() => setIsCaptureViewOpen(false)} />
        )}

        {view === 'result' && analysisResult && (
          <div className="w-full h-[85vh] overflow-y-auto custom-scrollbar relative">
            <ResultDashboard result={analysisResult} onReset={() => setView('home')} />
          </div>
        )}
      </main>

      <style>{`
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #7ec8a0; border-radius: 4px; }
        @keyframes scan-line { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
        .fade-up { animation: fade-up 0.6s ease forwards; }
        @keyframes fade-up { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
