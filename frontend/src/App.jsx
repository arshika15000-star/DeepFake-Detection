import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, UploadCloud, Camera, Eye, FileText, Layers, ShieldCheck, AlertTriangle, Mic, Type } from 'lucide-react';
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

  const handleFileUpload = (e, modality) => {
    const f = e.target.files[0];
    if (f) startAnalysis(f, modality);
  };

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ backgroundColor: '#000', color: '#fff' }}>
      <StarBackground />

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
          <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            
            {/* LEFT SIDE: 3D Visualization */}
            <div className="flex justify-center items-center">
              <div 
                className="relative w-80 h-96 border rounded-2xl flex items-center justify-center mesh-glow overflow-hidden" 
                style={{ backgroundColor: 'rgba(20, 20, 20, 0.5)', borderColor: 'rgba(126, 200, 160, 0.3)' }}
              >
                {/* Simulated 3D Head background */}
                <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full opacity-30" style={{ color: '#7ec8a0' }}>
                   <path fill="none" stroke="currentColor" strokeWidth="0.5" d="M30 20 Q50 0 70 20 T70 60 Q50 90 30 60 T30 20 M50 0 V90 M30 40 H70 M35 60 H65 M40 20 H60" />
                   <circle cx="50" cy="40" r="4" fill="currentColor" className="animate-pulse" />
                   <circle cx="35" cy="40" r="1.5" fill="currentColor" />
                   <circle cx="65" cy="40" r="1.5" fill="currentColor" />
                </svg>
                {/* Scanning line */}
                <div className="absolute top-0 left-0 right-0 h-1" style={{ background: '#7ec8a0', boxShadow: '0 0 15px #7ec8a0', animation: 'scan-line 3s linear infinite' }} />
              </div>
            </div>

            {/* RIGHT SIDE: Text and Actions */}
            <div className="text-left space-y-6">
              {/* Glowing Blur Behind Text */}
              <div className="absolute top-1/2 right-1/4 w-96 h-32 blur-[100px] pointer-events-none rounded-full" style={{ background: 'rgba(126, 200, 160, 0.25)' }} />

              <h1 className="text-5xl lg:text-7xl font-bold tracking-tight glow-text leading-tight">
                Deepfake Image & <br/> Video Detection
              </h1>
              
              <p className="text-lg text-gray-300 max-w-xl">
                Using Deep Learning for forensic analysis and digital evidence validation.
              </p>

              <div className="flex flex-wrap gap-4 pt-8">
                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => document.getElementById('video-upload').click()}
                >
                  <Play size={20} /> Analyze Video
                </button>
                <input id="video-upload" type="file" accept="video/*" className="hidden" onChange={(e) => handleFileUpload(e, 'video')} />
                
                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => document.getElementById('image-upload').click()}
                >
                  <UploadCloud size={20} /> Upload Image
                </button>
                <input id="image-upload" type="file" accept="image/*" className="hidden" onChange={(e) => handleFileUpload(e, 'image')} />

                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => document.getElementById('audio-upload').click()}
                >
                  <Mic size={20} /> Analyze Audio
                </button>
                <input id="audio-upload" type="file" accept="audio/*" className="hidden" onChange={(e) => handleFileUpload(e, 'audio')} />

                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => {
                    const txt = prompt("Paste your text for semantic AI analysis:");
                    if (txt) startAnalysis(txt, 'text');
                  }}
                >
                  <Type size={20} /> Verify Text
                </button>

                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => { setSelectedModality('image'); setIsCaptureViewOpen(true); }}
                >
                  <Camera size={20} /> Live Snapshot
                </button>
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
