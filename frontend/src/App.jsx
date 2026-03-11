import React, { useState, useRef, useEffect } from 'react';
import NeuralBackground from './components/NeuralBackground';
import NeuralCenterpiece from './components/NeuralCenterpiece';
import ModalityCard from './components/ModalityCard';
import ScannerView from './components/ScannerView';
import ResultDashboard from './components/ResultDashboard';
import { Camera, Video, Mic, Type, Upload, ArrowLeft, Activity } from 'lucide-react';
import axios from 'axios';

const API_BASE = "http://127.0.0.1:8000";

const modalities = [
  { id: 'image', title: 'Image Neural Scan', icon: <Camera />, desc: 'Detect facial warps, ELA compression, and frequency artifacts in static frames.', color: 'primary' },
  { id: 'video', title: 'Video Stream Probe', icon: <Video />, desc: 'Analyze temporal coherence, multi-frame consistency, and localized movement anomalies.', color: 'secondary' },
  { id: 'audio', title: 'Vocal Frequency Lab', icon: <Mic />, desc: 'Identify synthetic vocoder patterns, cloned speech jitter, and spectral floor gaps.', color: 'primary' },
  { id: 'text', title: 'Semantic Authenticity', icon: <Type />, desc: 'Evaluate linguistic formality, structural repetition, and LLM-specific syntactic markers.', color: 'secondary' },
];

export default function App() {
  const [view, setView] = useState('home'); // home, modality, scanning, result
  const [selectedModality, setSelectedModality] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [file, setFile] = useState(null);

  const handleModalitySelect = (mod) => {
    setSelectedModality(mod);
    setView('modality');
  };

  const startAnalysis = async (inputData) => {
    setView('scanning');

    try {
      const formData = new FormData();
      let endpoint = '';

      if (selectedModality.id === 'text') {
        endpoint = '/predict_text';
        formData.append('current_text', inputData);
      } else {
        endpoint = selectedModality.id === 'image' ? '/predict_image' :
          selectedModality.id === 'audio' ? '/predict_audio' : '/predict';
        formData.append('file', inputData);
      }

      const response = await axios.post(`${API_BASE}${endpoint}`, formData);
      setAnalysisResult(response.data);
    } catch (err) {
      console.error(err);
      alert("Neural Link Failure: " + (err.response?.data?.detail || err.message));
      setView('modality');
    }
  };

  const handleFileUpload = (e) => {
    const f = e.target.files[0];
    if (f) startAnalysis(f);
  };

  const handleTextSubmit = (txt) => {
    if (txt.trim()) startAnalysis(txt);
  };

  return (
    <div className="min-h-screen text-white overflow-x-hidden selection:bg-primary selection:text-dark">
      {/* <NeuralBackground /> */}

      {/* Navigation Header */}
      <nav className="fixed top-0 left-0 right-0 z-40 px-6 lg:px-12 py-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center bg-dark/20 backdrop-blur-3xl rounded-3xl p-4 border border-white/5 shadow-2xl">
          <div className="flex items-center gap-4 cursor-pointer" onClick={() => setView('home')}>
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-xl flex items-center justify-center font-black italic shadow-[0_0_15px_rgba(34,211,238,0.4)]">DT</div>
            <span className="text-2xl font-black italic tracking-tighter">DEEP<span className="text-primary tracking-normal not-italic font-light">TRUTH</span></span>
          </div>

          <div className="hidden md:flex gap-8 text-[10px] font-bold uppercase tracking-[0.3em] opacity-40">
            <span className="hover:text-primary hover:opacity-100 transition-all cursor-pointer">Protocol</span>
            <span className="hover:text-primary hover:opacity-100 transition-all cursor-pointer">Network</span>
            <span className="hover:text-primary hover:opacity-100 transition-all cursor-pointer">Archive</span>
          </div>

          <div className={`px-4 py-1.5 rounded-full border border-primary/20 text-[8px] font-black uppercase tracking-widest text-primary ${view === 'scanning' ? 'animate-pulse' : ''}`}>
            System Status: {view === 'scanning' ? 'Analyzing' : 'Ready'}
          </div>
        </div>
      </nav>

      {/* Main Views */}
      <main className="relative z-10 pt-32 px-6">
        {view === 'home' && (
          <div className="max-w-7xl mx-auto text-center space-y-20 py-12">
            <header className="space-y-6">
              {/* <NeuralCenterpiece /> */}
              <div className="inline-block px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-xs font-bold text-primary tracking-widest uppercase mb-4">
                SOTA Multimodal Deepfake Defense
              </div>
              <h1 className="text-7xl lg:text-9xl font-black italic tracking-tighter leading-[0.8]">
                EXPLAINABLE <br />
                <span className="text-gradient">NEURAL DEFENSE</span>
              </h1>
              <p className="max-w-2xl mx-auto text-dim text-lg leading-relaxed opacity-60">
                Unified forensic laboratory for detecting high-fidelity synthetic media using multimodal attention patterns and biometric signals.
              </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {modalities.map(mod => (
                <ModalityCard
                  key={mod.id}
                  title={mod.title}
                  icon={mod.icon}
                  description={mod.desc}
                  color={mod.color}
                  onClick={() => handleModalitySelect(mod)}
                />
              ))}
            </div>
          </div>
        )}

        {view === 'modality' && (
          <div className="max-w-3xl mx-auto py-20 space-y-12 animate-in slide-in-from-bottom-10 fade-in duration-500">
            <button
              onClick={() => setView('home')}
              className="flex items-center gap-3 text-xs font-bold uppercase tracking-widest text-dim hover:text-white transition-colors"
            >
              <ArrowLeft size={16} /> Back to Nexus
            </button>

            <div className="glass-morphism p-12 rounded-[3rem] border-primary/20 space-y-12">
              <div className="text-center space-y-4">
                <div className="text-6xl mb-4 italic flex justify-center">{selectedModality.icon}</div>
                <h2 className="text-4xl font-black italic tracking-tighter">{selectedModality.title}</h2>
                <p className="text-dim opacity-60">{selectedModality.desc}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div
                  className="group flex flex-col items-center justify-center bg-white/5 rounded-[2rem] p-10 border border-white/5 hover:border-primary/50 transition-all cursor-pointer relative overflow-hidden"
                  onClick={() => document.getElementById('file-upload').click()}
                >
                  <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Upload className="text-primary mb-4" size={40} />
                  <span className="text-sm font-black uppercase tracking-widest">Upload Deep File</span>
                  <input
                    id="file-upload"
                    type="file"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                </div>

                {selectedModality.id === 'text' ? (
                  <div className="bg-white/5 rounded-[2rem] p-6 border border-white/5">
                    <textarea
                      id="text-capture"
                      className="w-full h-32 bg-transparent border-none text-white font-mono text-sm resize-none focus:ring-0"
                      placeholder="PASTE SEMANTIC DATA HERE..."
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && e.ctrlKey) handleTextSubmit(e.target.value);
                      }}
                    />
                    <button
                      onClick={() => handleTextSubmit(document.getElementById('text-capture').value)}
                      className="w-full mt-4 bg-primary/10 border border-primary/30 py-4 rounded-2xl text-xs font-black uppercase tracking-widest hover:bg-primary hover:text-dark transition-all"
                    >
                      Execute Analysis
                    </button>
                  </div>
                ) : (
                  <div
                    className="group flex flex-col items-center justify-center bg-white/5 rounded-[2rem] p-10 border border-white/5 hover:border-secondary/50 transition-all cursor-pointer relative overflow-hidden"
                    onClick={() => alert(`Direct ${selectedModality.id} capture protocol active... (Simulation Mode)`)}
                  >
                    <div className="absolute inset-0 bg-secondary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                    <Activity className="text-secondary mb-4" size={40} />
                    <span className="text-sm font-black uppercase tracking-widest">Live Capture</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {view === 'scanning' && selectedModality && (
          <ScannerView
            modality={selectedModality.id}
            onComplete={() => setView('result')}
          />
        )}

        {view === 'result' && analysisResult && (
          <ResultDashboard
            result={analysisResult}
            onReset={() => setView('home')}
          />
        )}
      </main>
    </div>
  );
}
