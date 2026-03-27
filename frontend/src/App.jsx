import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, UploadCloud, Camera, Eye, FileText, Layers, ShieldCheck, AlertTriangle, Mic, Type, Link2, ArrowRight } from 'lucide-react';
import ScannerView from './components/ScannerView';
import CaptureView from './components/CaptureView';
import ResultDashboard from './components/ResultDashboard';

const getApiBaseUrl = () => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
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
        <div
          key={s.id}
          className="star"
          style={{
            left: `${s.left}%`,
            top: `${s.top}%`,
            width: s.size,
            height: s.size,
            animationDelay: `${s.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState('home'); // home, scanning, result
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isCaptureViewOpen, setIsCaptureViewOpen] = useState(false);
  const [selectedModality, setSelectedModality] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [urlInputs, setUrlInputs] = useState({ image: '', video: '', audio: '' });

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
        endpoint = modality === 'image' ? '/predict_image' : 
                   modality === 'audio' ? '/predict_audio' : '/predict';
        formData.append('file', inputData);
      }
      
      const response = await axios.post(`${API_BASE}${endpoint}`, formData);
      if (response.data.job_id) {
        pollJobStatus(response.data.job_id);
      } else {
        // Fallback if not using job_id
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

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (!f) return;
    
    // Auto-detect modality from mime type
    if (f.type.startsWith('image/')) handleFileUpload({ target: { files: [f] } }, 'image');
    else if (f.type.startsWith('video/')) handleFileUpload({ target: { files: [f] } }, 'video');
    else if (f.type.startsWith('audio/')) handleFileUpload({ target: { files: [f] } }, 'audio');
    else alert('❌ Unsupported file drop type. Please drop images, audio, or video files.');
  };

  return (
    <div 
      className="min-h-screen relative overflow-hidden" 
      style={{ backgroundColor: '#000', color: '#fff' }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <StarBackground />
      {isDragging && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm border-4 border-dashed border-[#7ec8a0]">
          <h2 className="text-4xl font-bold text-[#7ec8a0] animate-pulse">Drop File to Analyze</h2>
        </div>
      )}

      {/* ─── NAVBAR (Matching Reference Image) ─── */}
      <nav className="relative z-50 flex items-center justify-between px-10 py-6">
        {/* LOGO */}
        <div 
          className="flex items-center gap-4 cursor-pointer" 
          onClick={() => setView('home')}
        >
          <div className="w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold text-black" style={{ backgroundColor: '#7ec8a0', boxShadow: '0 0 20px rgba(126, 200, 160, 0.6)' }}>
            DF
          </div>
          <span className="text-3xl font-bold tracking-wide" style={{ color: '#7ec8a0', textShadow: '0 0 10px rgba(126, 200, 160, 0.4)' }}>
            DeepFake
          </span>
        </div>

        {/* LINKS */}
        <div className="hidden md:flex items-center gap-8">
          <div className="nav-link active">Home</div>
          <div className="nav-link">About</div>
          <div className="nav-link">Login or Signup</div>
          <div className="nav-link">Contact Us</div>
        </div>
      </nav>

      {/* ─── MAIN CONTENT ─── */}
      <main className="relative z-10 w-full h-[85vh] flex items-center justify-center px-10">
        
        {view === 'home' && (
          <div className="w-full max-w-[90vw] flex flex-col items-center mt-10 space-y-12">
            
            <div className="text-center space-y-4">
              <h1 className="text-5xl lg:text-7xl font-bold tracking-tight glow-text leading-tight">
                Deepfake Intelligence
              </h1>
              <p className="text-xl text-gray-300 max-w-2xl mx-auto">
                Select a modality to begin forensic analysis. Upload files or capture live data.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8 w-full pb-20">
              
              {/* IMAGE COLUMN */}
              <div className="glass-morphism rounded-3xl p-8 flex flex-col items-center text-center border transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(126,200,160,0.3)] shadow-xl relative overflow-hidden group" style={{ borderColor: 'rgba(126, 200, 160, 0.4)' }}>
                <div className="absolute inset-0 bg-gradient-to-b from-[#7ec8a0]/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <UploadCloud size={48} className="text-[#7ec8a0] mb-4" />
                <h2 className="text-2xl font-bold mb-2">Image</h2>
                <p className="text-sm text-gray-400 mb-8 flex-1">
                  Detect AI-generated faces, GAN artifacts, and manipulated pixels in photos.
                </p>
                <div className="w-full space-y-3 z-10">
                  <button onClick={() => document.getElementById('image-upload').click()} className="w-full py-3 rounded-xl font-bold bg-[#7ec8a0] text-black hover:bg-[#6ab38c] transition-colors flex items-center justify-center gap-2">
                    <UploadCloud size={18} /> Upload Image
                  </button>
                  <input id="image-upload" type="file" accept="image/*" className="hidden" onChange={(e) => handleFileUpload(e, 'image')} />
                  <button onClick={() => { setSelectedModality('image'); setIsCaptureViewOpen(true); }} className="w-full py-3 rounded-xl font-bold border border-[#7ec8a0] text-[#7ec8a0] hover:bg-[#7ec8a0]/10 transition-colors flex items-center justify-center gap-2">
                    <Camera size={18} /> Live Snapshot
                  </button>
                  <div className="flex gap-2">
                    <input
                      id="image-url-input"
                      type="url"
                      placeholder="Paste image URL…"
                      value={urlInputs.image}
                      onChange={e => setUrlInputs(p => ({ ...p, image: e.target.value }))}
                      className="flex-1 px-3 py-2 rounded-xl text-sm bg-black/40 border border-[#7ec8a0]/30 text-white placeholder-gray-500 focus:outline-none focus:border-[#7ec8a0]"
                    />
                    <button onClick={() => handleUrlAnalysis('image')} className="px-3 py-2 rounded-xl bg-[#7ec8a0]/20 border border-[#7ec8a0]/40 text-[#7ec8a0] hover:bg-[#7ec8a0]/30 transition-colors" title="Analyze URL">
                      <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              </div>

              {/* VIDEO COLUMN */}
              <div className="glass-morphism rounded-3xl p-8 flex flex-col items-center text-center border transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(99,144,255,0.3)] shadow-xl relative overflow-hidden group" style={{ borderColor: 'rgba(99, 144, 255, 0.4)' }}>
                <div className="absolute inset-0 bg-gradient-to-b from-[#6390ff]/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <Play size={48} className="text-[#6390ff] mb-4" />
                <h2 className="text-2xl font-bold mb-2">Video</h2>
                <p className="text-sm text-gray-400 mb-8 flex-1">
                  Analyze temporal coherence, rPPG pulse, and frame-by-frame deepfake signatures.
                </p>
                <div className="w-full space-y-3 z-10">
                  <button onClick={() => document.getElementById('video-upload').click()} className="w-full py-3 rounded-xl font-bold bg-[#6390ff] text-white hover:bg-[#4a72d1] transition-colors flex items-center justify-center gap-2">
                    <UploadCloud size={18} /> Upload Video
                  </button>
                  <input id="video-upload" type="file" accept="video/*" className="hidden" onChange={(e) => handleFileUpload(e, 'video')} />
                  <button onClick={() => { setSelectedModality('video'); setIsCaptureViewOpen(true); }} className="w-full py-3 rounded-xl font-bold border border-[#6390ff] text-[#6390ff] hover:bg-[#6390ff]/10 transition-colors flex items-center justify-center gap-2">
                    <Camera size={18} /> Record Video
                  </button>
                  <div className="flex gap-2">
                    <input
                      id="video-url-input"
                      type="url"
                      placeholder="Paste video URL…"
                      value={urlInputs.video}
                      onChange={e => setUrlInputs(p => ({ ...p, video: e.target.value }))}
                      className="flex-1 px-3 py-2 rounded-xl text-sm bg-black/40 border border-[#6390ff]/30 text-white placeholder-gray-500 focus:outline-none focus:border-[#6390ff]"
                    />
                    <button onClick={() => handleUrlAnalysis('video')} className="px-3 py-2 rounded-xl bg-[#6390ff]/20 border border-[#6390ff]/40 text-[#6390ff] hover:bg-[#6390ff]/30 transition-colors" title="Analyze URL">
                      <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              </div>

              {/* AUDIO COLUMN */}
              <div className="glass-morphism rounded-3xl p-8 flex flex-col items-center text-center border transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(255,144,99,0.3)] shadow-xl relative overflow-hidden group" style={{ borderColor: 'rgba(255, 144, 99, 0.4)' }}>
                 <div className="absolute inset-0 bg-gradient-to-b from-[#ff9063]/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <Mic size={48} className="text-[#ff9063] mb-4" />
                <h2 className="text-2xl font-bold mb-2">Audio</h2>
                <p className="text-sm text-gray-400 mb-8 flex-1">
                  Extract acoustic fingerprints and detect cloned or synthetic AI voices.
                </p>
                <div className="w-full space-y-3 z-10">
                  <button onClick={() => document.getElementById('audio-upload').click()} className="w-full py-3 rounded-xl font-bold bg-[#ff9063] text-white hover:bg-[#e07548] transition-colors flex items-center justify-center gap-2">
                    <UploadCloud size={18} /> Upload Audio
                  </button>
                  <input id="audio-upload" type="file" accept="audio/*" className="hidden" onChange={(e) => handleFileUpload(e, 'audio')} />
                  <button onClick={() => { setSelectedModality('audio'); setIsCaptureViewOpen(true); }} className="w-full py-3 rounded-xl font-bold border border-[#ff9063] text-[#ff9063] hover:bg-[#ff9063]/10 transition-colors flex items-center justify-center gap-2">
                    <Mic size={18} /> Record Audio
                  </button>
                  <div className="flex gap-2">
                    <input
                      id="audio-url-input"
                      type="url"
                      placeholder="Paste audio URL…"
                      value={urlInputs.audio}
                      onChange={e => setUrlInputs(p => ({ ...p, audio: e.target.value }))}
                      className="flex-1 px-3 py-2 rounded-xl text-sm bg-black/40 border border-[#ff9063]/30 text-white placeholder-gray-500 focus:outline-none focus:border-[#ff9063]"
                    />
                    <button onClick={() => handleUrlAnalysis('audio')} className="px-3 py-2 rounded-xl bg-[#ff9063]/20 border border-[#ff9063]/40 text-[#ff9063] hover:bg-[#ff9063]/30 transition-colors" title="Analyze URL">
                      <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              </div>

              {/* TEXT COLUMN */}
              <div className="glass-morphism rounded-3xl p-8 flex flex-col items-center text-center border transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(200,126,255,0.3)] shadow-xl relative overflow-hidden group" style={{ borderColor: 'rgba(200, 126, 255, 0.4)' }}>
                <div className="absolute inset-0 bg-gradient-to-b from-[#c87eff]/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <Type size={48} className="text-[#c87eff] mb-4" />
                <h2 className="text-2xl font-bold mb-2">Text</h2>
                <p className="text-sm text-gray-400 mb-8 flex-1">
                  Evaluate linguistic syntax, perplexity, and AI generator writing patterns.
                </p>
                <div className="w-full space-y-3 z-10">
                  <button onClick={() => {
                      const txt = prompt("Paste your text for semantic AI analysis:");
                      if (txt) startAnalysis(txt, 'text');
                    }} className="w-full py-3 rounded-xl font-bold bg-[#c87eff] text-white hover:bg-[#a55cd9] transition-colors flex items-center justify-center gap-2">
                    <Type size={18} /> Paste Text
                  </button>
                  <button onClick={() => {}} className="w-full py-3 rounded-xl font-bold border border-[#c87eff] text-[#c87eff] opacity-50 cursor-not-allowed flex items-center justify-center gap-2">
                    <FileText size={18} /> Upload Document (Coming Soon)
                  </button>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* ─── SCANNING & CAPTURE & RESULTS ─── */}
        {view === 'scanning' && selectedModality && (
          <ScannerView
            modality={selectedModality}
            progress={progress}
          />
        )}

        {isCaptureViewOpen && (
          <CaptureView
            modality={selectedModality}
            onCapture={(file) => {
              setIsCaptureViewOpen(false);
              startAnalysis(file, selectedModality);
            }}
            onClose={() => setIsCaptureViewOpen(false)}
          />
        )}

        {view === 'result' && analysisResult && (
            <div className="w-full h-[85vh] overflow-y-auto custom-scrollbar relative">
                <ResultDashboard result={analysisResult} onReset={() => setView('home')} />
            </div>
        )}
      </main>
      
      {/* GLOBAL KEYFRAMES */}
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
