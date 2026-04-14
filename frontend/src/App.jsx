import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Play, UploadCloud, Camera, Mic, Type, ArrowRight, X,
  Mail, User, Lock, MessageSquare, Send, Eye, EyeOff,
  Shield, Brain, Layers, Zap, ChevronRight, LogOut, Info
} from 'lucide-react';
import ScannerView from './components/ScannerView';
import CaptureView from './components/CaptureView';
import ResultDashboard from './components/ResultDashboard';
import LandingContent from './components/LandingContent';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import AIAssistant from './components/AIAssistant';
const getApiBaseUrl = () => {
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  return "http://127.0.0.1:8005";
};
const API_BASE = getApiBaseUrl();

// ─── GOOGLE SVG ICON ───────────────────────────────────────────────────────
function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" fill="none">
      <path d="M44.5 20H24v8.5h11.8C34.7 33.9 30.1 37 24 37c-7.2 0-13-5.8-13-13s5.8-13 13-13c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 11.8 2 2 11.8 2 24s9.8 22 22 22c11 0 21-8 21-22 0-1.3-.2-2.7-.5-4z" fill="#FFC107"/>
      <path d="M6.3 14.7l7 5.1C15.2 16.5 19.3 14 24 14c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 16.3 2 9.7 7.4 6.3 14.7z" fill="#FF3D00"/>
      <path d="M24 46c5.5 0 10.5-2.1 14.2-5.4l-6.6-5.6C29.7 36.6 27 37.5 24 37.5c-6 0-11.1-4-13-9.5l-7 5.4C7.4 41.3 15.1 46 24 46z" fill="#4CAF50"/>
      <path d="M44.5 20H24v8.5h11.8c-.5 2.4-2 4.5-4.1 6l6.6 5.6C42.2 36.6 45 30.7 45 24c0-1.3-.2-2.7-.5-4z" fill="#1976D2"/>
    </svg>
  );
}

// ─── STAR BACKGROUND ───────────────────────────────────────────────────────
function StarBackground() {
  const stars = React.useMemo(() => Array.from({ length: 70 }, (_, i) => ({
    id: i, left: Math.random() * 100, top: Math.random() * 100,
    size: Math.random() * 3 + 1, delay: Math.random() * 5,
    tx1: Math.random() * 150 - 75, ty1: Math.random() * 150 - 75,
    tx2: Math.random() * 150 - 75, ty2: Math.random() * 150 - 75,
    tx3: Math.random() * 150 - 75, ty3: Math.random() * 150 - 75,
    duration: Math.random() * 40 + 40
  })), []);

  return (
    <div className="star-bg">
      <style>{`
        @keyframes float-dust {
          0%, 100% { transform: translate(0px, 0px); }
          25% { transform: translate(var(--tx1), var(--ty1)); }
          50% { transform: translate(var(--tx2), var(--ty2)); }
          75% { transform: translate(var(--tx3), var(--ty3)); }
        }
      `}</style>
      {stars.map(s => (
        <div key={s.id} className="star"
          style={{ 
            left: `${s.left}%`, top: `${s.top}%`, 
            width: s.size, height: s.size, 
            '--tx1': `${s.tx1}px`, '--ty1': `${s.ty1}px`,
            '--tx2': `${s.tx2}px`, '--ty2': `${s.ty2}px`,
            '--tx3': `${s.tx3}px`, '--ty3': `${s.ty3}px`,
            animation: `twinkle ${s.delay + 3}s ease-in-out infinite alternate, float-dust ${s.duration}s ease-in-out infinite`
          }} />
      ))}
    </div>
  );
}

// ─── THEME TOGGLE ──────────────────────────────────────────────────────────
function ThemeToggle({ isDark, onToggle }) {
  return (
    <button id="theme-toggle-btn" onClick={onToggle}
      className={`theme-toggle-btn ${isDark ? 'dark' : 'light'}`}
      title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'} aria-label="Toggle theme">
      <span className="toggle-knob">{isDark ? '🌙' : '☀️'}</span>
    </button>
  );
}

// ─── ABOUT MODAL ───────────────────────────────────────────────────────────
function AboutModal({ onClose, isDark }) {
  const bg = isDark ? 'rgba(8,8,18,0.98)' : 'rgba(248,250,252,0.98)';
  const sub = isDark ? '#94a3b8' : '#64748b';
  const cardBg = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)';
  const borderCol = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';

  const modalities = [
    { icon: UploadCloud, color: '#7ec8a0', label: 'Image Analysis',
      desc: 'EfficientNet-B4 + XceptionNet detect GAN artifacts, face-swap boundaries, and pixel-level inconsistencies in photos.' },
    { icon: Play, color: '#6390ff', label: 'Video Analysis',
      desc: 'Frame-by-frame temporal analysis with rPPG pulse detection, optical flow, and BlazeFace landmark tracking.' },
    { icon: Mic, color: '#ff9063', label: 'Audio Analysis',
      desc: 'Wav2Vec 2.0 extracts mel-spectrogram features to detect voice cloning, TTS synthesis, and acoustic manipulation.' },
    { icon: Type, color: '#c87eff', label: 'Text Analysis',
      desc: 'BERT-based perplexity scoring and n-gram entropy analysis to identify AI-generated text patterns.' },
  ];

  const stats = [
    { value: '4', label: 'Modalities' },
    { value: '97%+', label: 'Accuracy' },
    { value: 'XAI', label: 'Explainable' },
    { value: 'Real-time', label: 'Analysis' },
  ];

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/75 backdrop-blur-md p-4 overflow-y-auto">
      <div className="relative w-full max-w-3xl rounded-3xl shadow-2xl fade-up my-4"
        style={{ background: bg, border: '1px solid rgba(126,200,160,0.25)' }}>

        {/* Close */}
        <button onClick={onClose}
          className="absolute top-5 right-5 z-10 p-2 rounded-full hover:bg-white/10 transition-colors"
          style={{ color: '#7ec8a0' }}>
          <X size={20} />
        </button>

        {/* Hero Header */}
        <div className="relative p-10 pb-6 text-center overflow-hidden rounded-t-3xl"
          style={{ background: 'linear-gradient(135deg, rgba(126,200,160,0.12) 0%, rgba(99,144,255,0.10) 50%, rgba(200,126,255,0.08) 100%)' }}>
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-2xl font-black text-black"
              style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)', boxShadow: '0 0 30px rgba(126,200,160,0.5)' }}>
              DF
            </div>
          </div>
          <h2 className="text-3xl font-black mb-2" style={{ color: '#7ec8a0' }}>
            DeepFake Detection Platform
          </h2>
          <p className="text-base max-w-xl mx-auto" style={{ color: sub }}>
            A multimodal, explainable AI system built to detect synthetic manipulations across images, videos, audio, and text — in real time.
          </p>

          {/* Stats Row */}
          <div className="flex items-center justify-center gap-6 mt-6 flex-wrap">
            {stats.map(s => (
              <div key={s.label} className="text-center px-4 py-2 rounded-2xl"
                style={{ background: cardBg, border: `1px solid ${borderCol}` }}>
                <div className="text-2xl font-black" style={{ color: '#7ec8a0' }}>{s.value}</div>
                <div className="text-xs font-semibold uppercase tracking-wider mt-0.5" style={{ color: sub }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Body */}
        <div className="p-8 space-y-8">

          {/* What are Deepfakes */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Shield size={18} style={{ color: '#7ec8a0' }} />
              <h3 className="text-lg font-bold" style={{ color: isDark ? '#fff' : '#0f172a' }}>What Are Deepfakes?</h3>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: sub }}>
              Deepfakes are AI-generated synthetic media — fabricated images, videos, audio recordings, and text — created using deep learning techniques like GANs (Generative Adversarial Networks), diffusion models, and large language models. They pose serious risks to personal privacy, misinformation, and national security. Our platform uses state-of-the-art forensic AI to expose these manipulations.
            </p>
          </section>

          {/* Modalities Grid */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Layers size={18} style={{ color: '#6390ff' }} />
              <h3 className="text-lg font-bold" style={{ color: isDark ? '#fff' : '#0f172a' }}>Multimodal Detection Engine</h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {modalities.map(m => (
                <div key={m.label} className="flex items-start gap-3 p-4 rounded-2xl"
                  style={{ background: cardBg, border: `1px solid ${borderCol}` }}>
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: `${m.color}18`, border: `1px solid ${m.color}40` }}>
                    <m.icon size={18} style={{ color: m.color }} />
                  </div>
                  <div>
                    <div className="text-sm font-bold mb-1" style={{ color: m.color }}>{m.label}</div>
                    <div className="text-xs leading-relaxed" style={{ color: sub }}>{m.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* How it works */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <Brain size={18} style={{ color: '#c87eff' }} />
              <h3 className="text-lg font-bold" style={{ color: isDark ? '#fff' : '#0f172a' }}>How It Works</h3>
            </div>
            <div className="flex flex-col gap-2">
              {[
                { step: '01', title: 'Upload or Capture', desc: 'Submit any image, video, audio file, or text — via file upload, live camera/mic capture, or URL.' },
                { step: '02', title: 'Multimodal Fusion', desc: 'Each modality is independently analyzed by a specialized neural network and scores are fused.' },
                { step: '03', title: 'Explainable Results', desc: 'GradCAM heatmaps, confidence scores, and per-feature breakdowns make every verdict transparent.' },
              ].map((s, i) => (
                <div key={i} className="flex items-start gap-4 p-4 rounded-2xl"
                  style={{ background: cardBg, border: `1px solid ${borderCol}` }}>
                  <div className="text-2xl font-black flex-shrink-0" style={{ color: 'rgba(126,200,160,0.3)' }}>{s.step}</div>
                  <div>
                    <div className="text-sm font-bold mb-0.5" style={{ color: isDark ? '#e2e8f0' : '#1e293b' }}>{s.title}</div>
                    <div className="text-xs leading-relaxed" style={{ color: sub }}>{s.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Technology Stack */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Zap size={18} style={{ color: '#ff9063' }} />
              <h3 className="text-lg font-bold" style={{ color: isDark ? '#fff' : '#0f172a' }}>Technology Stack</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {['EfficientNet-B4', 'XceptionNet', 'Wav2Vec 2.0', 'BERT', 'GradCAM XAI',
                'BlazeFace', 'Python / FastAPI', 'React + Vite', 'PyTorch'].map(t => (
                <span key={t} className="px-3 py-1 rounded-full text-xs font-bold"
                  style={{ background: 'rgba(126,200,160,0.12)', color: '#7ec8a0', border: '1px solid rgba(126,200,160,0.25)' }}>
                  {t}
                </span>
              ))}
            </div>
          </section>

          {/* Project Info */}
          <div className="p-4 rounded-2xl text-center"
            style={{ background: 'linear-gradient(135deg, rgba(126,200,160,0.08), rgba(99,144,255,0.06))', border: '1px solid rgba(126,200,160,0.2)' }}>
            <p className="text-xs font-semibold" style={{ color: sub }}>
              🎓 <strong style={{ color: '#7ec8a0' }}>Academic Research Project</strong> — Multimodal &amp; Explainable AI for Deepfake Detection
            </p>
            <p className="text-xs mt-1" style={{ color: sub }}>
              Built with ❤️ using state-of-the-art deep learning research (2024–2025)
            </p>
          </div>

          {/* Back to Home Button */}
          <div className="flex justify-center mt-6">
            <button
              onClick={onClose}
              className="px-8 py-3 rounded-xl font-bold transition-all hover:scale-105 active:scale-95 flex items-center gap-2"
              style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)', color: '#000' }}
            >
              ← Back to Home
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── FORGOT PASSWORD MODAL ─────────────────────────────────────────────────
function ForgotPasswordModal({ onClose, isDark }) {
  const [email, setEmail] = useState('');
  const [step, setStep] = useState('enter_email');
  const [msg, setMsg] = useState('');
  const bg = isDark ? 'rgba(15,15,25,0.97)' : 'rgba(255,255,255,0.97)';
  const sub = isDark ? '#94a3b8' : '#64748b';
  const inp = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)';
  const inpCol = isDark ? '#fff' : '#0f172a';

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!localStorage.getItem(`df_user_${email}`)) { setMsg('❌ No account found with this email.'); return; }
    setStep('sent');
  };

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="auth-modal-inner relative w-full max-w-sm rounded-3xl p-8 shadow-2xl fade-up"
        style={{ background: bg, border: '1px solid rgba(126,200,160,0.3)' }}>
        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 transition-colors" style={{ color: '#7ec8a0' }}>
          <X size={18} />
        </button>
        {step === 'enter_email' ? (
          <>
            <div className="text-center mb-6">
              <div className="w-14 h-14 rounded-full flex items-center justify-center mx-auto mb-3"
                style={{ background: 'rgba(126,200,160,0.15)', border: '1px solid rgba(126,200,160,0.3)' }}>
                <Lock size={24} style={{ color: '#7ec8a0' }} />
              </div>
              <h2 className="text-2xl font-bold" style={{ color: '#7ec8a0' }}>Forgot Password?</h2>
              <p className="text-sm mt-1" style={{ color: sub }}>Enter your email and we'll send a reset link.</p>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input required type="email" placeholder="Your email address" value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="w-full pl-9 pr-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors border"
                  style={{ background: inp, borderColor: 'rgba(126,200,160,0.2)', color: inpCol }} />
              </div>
              {msg && <p className="text-xs text-center font-medium py-2 px-3 rounded-lg"
                style={{ background: 'rgba(251,113,133,0.1)', color: '#fb7185' }}>{msg}</p>}
              <button type="submit" id="forgot-send-btn"
                className="w-full py-3 rounded-xl font-bold text-black transition-all hover:scale-105"
                style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)' }}>
                Send Reset Link
              </button>
              <button type="button" onClick={onClose}
                className="w-full py-2 text-sm font-semibold transition-all hover:opacity-80"
                style={{ color: '#7ec8a0' }}>← Back to Login</button>
            </form>
          </>
        ) : (
          <div className="text-center py-6 fade-up">
            <div className="text-5xl mb-4">📧</div>
            <h3 className="text-xl font-bold mb-2" style={{ color: '#7ec8a0' }}>Check Your Inbox!</h3>
            <p className="text-sm mb-6" style={{ color: sub }}>
              A password reset link has been sent to <strong style={{ color: '#7ec8a0' }}>{email}</strong>.<br />
              Check your spam folder if you don't see it.
            </p>
            <button onClick={onClose}
              className="px-6 py-2 rounded-xl font-bold text-black transition-all hover:scale-105"
              style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)' }}>Done</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── LOGIN / SIGNUP MODAL ──────────────────────────────────────────────────
function AuthModal({ onClose, onForgotPassword, isDark, onLogin }) {
  const [tab, setTab] = useState('login');
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [showPwd, setShowPwd] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [msg, setMsg] = useState('');
  const [googleLoading, setGoogleLoading] = useState(false);

  const GOOGLE_ACCOUNTS = [
    { name: 'Vanshika Saxena', email: 'vanshika@gmail.com', avatar: 'VS', provider: 'google' },
    { name: 'Google User', email: 'user@gmail.com', avatar: 'GU', provider: 'google' },
  ];

  const handleGoogleSignIn = () => {
    setGoogleLoading(true);
    setTimeout(() => {
      // Pick first Google account as demo (simulates real OAuth session)
      const user = GOOGLE_ACCOUNTS[0];
      localStorage.setItem('df_current_user', JSON.stringify(user));
      setMsg(`✅ Signed in as ${user.name}!`);
      setGoogleLoading(false);
      setTimeout(() => { onLogin(user); onClose(); }, 900);
    }, 1400);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (tab === 'signup' && form.password !== form.confirm) { setMsg('❌ Passwords do not match.'); return; }
    const key = `df_user_${form.email}`;
    if (tab === 'signup') {
      if (localStorage.getItem(key)) { setMsg('❌ Email already registered.'); return; }
      const user = { name: form.name, email: form.email, password: form.password, avatar: form.name.slice(0, 2).toUpperCase() };
      localStorage.setItem(key, JSON.stringify(user));
      setMsg('✅ Account created! Logging you in…');
      setTimeout(() => {
        localStorage.setItem('df_current_user', JSON.stringify(user));
        onLogin(user);
        onClose();
      }, 900);
    } else {
      const stored = localStorage.getItem(key);
      if (!stored) { setMsg('❌ Account not found. Please sign up first.'); return; }
      const user = JSON.parse(stored);
      if (user.password !== form.password) { setMsg('❌ Incorrect password.'); return; }
      localStorage.setItem('df_current_user', JSON.stringify(user));
      setMsg(`✅ Welcome back, ${user.name || user.email}!`);
      setTimeout(() => { onLogin(user); onClose(); }, 900);
    }
  };

  const bg = isDark ? 'rgba(15,15,25,0.97)' : 'rgba(255,255,255,0.97)';
  const inp = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)';
  const inpCol = isDark ? '#fff' : '#0f172a';
  const sub = isDark ? '#94a3b8' : '#64748b';

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="auth-modal-inner relative w-full max-w-md rounded-3xl p-8 shadow-2xl fade-up"
        style={{ background: bg, border: '1px solid rgba(126,200,160,0.3)' }}>
        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 transition-colors" style={{ color: '#7ec8a0' }}>
          <X size={18} />
        </button>
        <h2 className="text-2xl font-bold mb-1 text-center" style={{ color: '#7ec8a0' }}>
          {tab === 'login' ? 'Welcome Back' : 'Create Account'}
        </h2>
        <p className="text-center text-sm mb-6" style={{ color: sub }}>DeepFake Detection Platform</p>

        {/* Tabs */}
        <div className="flex rounded-xl overflow-hidden border mb-5" style={{ borderColor: 'rgba(126,200,160,0.2)' }}>
          {['login', 'signup'].map(t => (
            <button key={t} onClick={() => { setTab(t); setMsg(''); }}
              className="flex-1 py-2 text-sm font-bold transition-all"
              style={{ background: tab === t ? 'rgba(126,200,160,0.18)' : 'transparent', color: tab === t ? '#7ec8a0' : sub }}>
              {t === 'login' ? 'Log In' : 'Sign Up'}
            </button>
          ))}
        </div>

        {/* Google Sign In */}
        <button className="google-btn mb-2" onClick={handleGoogleSignIn} disabled={googleLoading} id="google-signin-btn">
          {googleLoading ? <span style={{ fontSize: 18 }}>⟳</span> : <GoogleIcon />}
          {googleLoading ? 'Signing in with Google…' : 'Continue with Google'}
        </button>

        <div className="auth-divider">or</div>

        <form onSubmit={handleSubmit} className="space-y-4 mt-3">
          {tab === 'signup' && (
            <div className="relative">
              <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input required placeholder="Full Name" value={form.name}
                onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                className="w-full pl-9 pr-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors border"
                style={{ background: inp, borderColor: 'rgba(126,200,160,0.2)', color: inpCol }} />
            </div>
          )}
          <div className="relative">
            <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input required type="email" placeholder="Email Address" value={form.email}
              onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
              className="w-full pl-9 pr-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors border"
              style={{ background: inp, borderColor: 'rgba(126,200,160,0.2)', color: inpCol }} />
          </div>
          <div className="relative">
            <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input required type={showPwd ? 'text' : 'password'} placeholder="Password" value={form.password}
              onChange={e => setForm(p => ({ ...p, password: e.target.value }))}
              className="w-full pl-9 pr-10 py-3 rounded-xl text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors border"
              style={{ background: inp, borderColor: 'rgba(126,200,160,0.2)', color: inpCol }} />
            <button type="button" onClick={() => setShowPwd(v => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-[#7ec8a0] transition-colors">
              {showPwd ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
          </div>
          {tab === 'login' && (
            <button type="button" className="forgot-link" onClick={onForgotPassword} id="forgot-password-btn">
              Forgot password?
            </button>
          )}
          {tab === 'signup' && (
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input required type={showConfirm ? 'text' : 'password'} placeholder="Confirm Password" value={form.confirm}
                onChange={e => setForm(p => ({ ...p, confirm: e.target.value }))}
                className="w-full pl-9 pr-10 py-3 rounded-xl text-sm focus:outline-none focus:border-[#7ec8a0] transition-colors border"
                style={{ background: inp, borderColor: 'rgba(126,200,160,0.2)', color: inpCol }} />
              <button type="button" onClick={() => setShowConfirm(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-[#7ec8a0] transition-colors">
                {showConfirm ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
          )}
          {msg && (
            <p className="text-xs text-center font-medium py-2 px-3 rounded-lg"
              style={{ background: msg.startsWith('✅') ? 'rgba(126,200,160,0.1)' : 'rgba(251,113,133,0.1)', color: msg.startsWith('✅') ? '#7ec8a0' : '#fb7185' }}>
              {msg}
            </p>
          )}
          <button type="submit" id={tab === 'login' ? 'login-submit-btn' : 'signup-submit-btn'}
            className="w-full py-3 rounded-xl font-bold text-black transition-all hover:scale-105 active:scale-95 mt-1"
            style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)' }}>
            {tab === 'login' ? 'Log In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  );
}

// ─── CONTACT MODAL ─────────────────────────────────────────────────────────
function ContactModal({ onClose, isDark }) {
  const [form, setForm] = useState({ name: '', email: '', subject: '', message: '' });
  const [sent, setSent] = useState(false);
  const bg = isDark ? 'rgba(15,15,25,0.97)' : 'rgba(255,255,255,0.97)';
  const inp = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)';
  const inpCol = isDark ? '#fff' : '#0f172a';
  const sub = isDark ? '#9ca3af' : '#64748b';

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="auth-modal-inner relative w-full max-w-lg rounded-3xl p-8 shadow-2xl fade-up"
        style={{ background: bg, border: '1px solid rgba(99,144,255,0.3)' }}>
        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full hover:bg-white/10 transition-colors" style={{ color: '#6390ff' }}>
          <X size={18} />
        </button>
        <h2 className="text-2xl font-bold mb-1 text-center" style={{ color: '#6390ff' }}>Contact Us</h2>
        <p className="text-center text-sm mb-6" style={{ color: sub }}>We'd love to hear from you. Drop us a message!</p>
        {sent ? (
          <div className="text-center py-12 fade-up">
            <div className="text-5xl mb-4">✅</div>
            <h3 className="text-xl font-bold text-[#6390ff] mb-2">Message Sent!</h3>
            <p className="text-sm" style={{ color: sub }}>Thank you for reaching out. We'll get back to you soon.</p>
          </div>
        ) : (
          <form onSubmit={e => { e.preventDefault(); setSent(true); setTimeout(onClose, 2500); }} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="relative">
                <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input required placeholder="Your Name" value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                  className="w-full pl-9 pr-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#6390ff] transition-colors border"
                  style={{ background: inp, borderColor: 'rgba(99,144,255,0.2)', color: inpCol }} />
              </div>
              <div className="relative">
                <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input required type="email" placeholder="Email" value={form.email} onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
                  className="w-full pl-9 pr-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#6390ff] transition-colors border"
                  style={{ background: inp, borderColor: 'rgba(99,144,255,0.2)', color: inpCol }} />
              </div>
            </div>
            <div className="relative">
              <MessageSquare size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input required placeholder="Subject" value={form.subject} onChange={e => setForm(p => ({ ...p, subject: e.target.value }))}
                className="w-full pl-9 pr-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#6390ff] transition-colors border"
                style={{ background: inp, borderColor: 'rgba(99,144,255,0.2)', color: inpCol }} />
            </div>
            <textarea required rows={4} placeholder="Your message…" value={form.message} onChange={e => setForm(p => ({ ...p, message: e.target.value }))}
              className="w-full px-4 py-3 rounded-xl text-sm focus:outline-none focus:border-[#6390ff] transition-colors resize-none border"
              style={{ background: inp, borderColor: 'rgba(99,144,255,0.2)', color: inpCol }} />
            <button type="submit" id="contact-submit-btn"
              className="w-full py-3 rounded-xl font-bold text-white flex items-center justify-center gap-2 transition-all hover:scale-105"
              style={{ background: 'linear-gradient(135deg,#6390ff,#4a72d1)' }}>
              <Send size={16} /> Send Message
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

// ─── USER AVATAR / NAV PROFILE ─────────────────────────────────────────────
function UserNav({ user, onLogout, isDark }) {
  const [open, setOpen] = useState(false);
  const initials = user.avatar || (user.name ? user.name.slice(0, 2).toUpperCase() : user.email.slice(0, 2).toUpperCase());
  const sub = isDark ? '#94a3b8' : '#64748b';
  const menuBg = isDark ? 'rgba(15,15,25,0.97)' : 'rgba(255,255,255,0.97)';

  return (
    <div className="relative">
      <button onClick={() => setOpen(v => !v)} id="user-avatar-btn"
        className="flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all hover:scale-105"
        style={{ borderColor: 'rgba(126,200,160,0.4)', background: 'rgba(126,200,160,0.08)' }}>
        <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-black text-black"
          style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)' }}>
          {initials}
        </div>
        <span className="text-sm font-semibold max-w-[100px] truncate" style={{ color: '#7ec8a0' }}>
          {user.name || user.email}
        </span>
        <ChevronRight size={12} style={{ color: '#7ec8a0', transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-56 rounded-2xl shadow-2xl z-[300] overflow-hidden fade-up"
          style={{ background: menuBg, border: '1px solid rgba(126,200,160,0.2)' }}>
          <div className="px-4 py-3 border-b" style={{ borderColor: 'rgba(126,200,160,0.1)' }}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full flex items-center justify-center font-black text-black"
                style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)' }}>{initials}</div>
              <div className="overflow-hidden">
                <div className="text-sm font-bold truncate" style={{ color: isDark ? '#fff' : '#0f172a' }}>{user.name || 'User'}</div>
                <div className="text-xs truncate" style={{ color: sub }}>{user.email}</div>
                {user.provider === 'google' && (
                  <div className="flex items-center gap-1 mt-0.5">
                    <GoogleIcon />
                    <span className="text-[10px]" style={{ color: sub }}>Google Account</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          <button onClick={() => { setOpen(false); onLogout(); }} id="logout-btn"
            className="w-full flex items-center gap-2 px-4 py-3 text-sm font-semibold hover:bg-red-500/10 transition-colors"
            style={{ color: '#fb7185' }}>
            <LogOut size={15} /> Sign Out
          </button>
        </div>
      )}
    </div>
  );
}

// ─── MAIN APP ──────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState('home');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isCaptureViewOpen, setIsCaptureViewOpen] = useState(false);
  const [selectedModality, setSelectedModality] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [urlInputs, setUrlInputs] = useState({ image: '', video: '', audio: '' });
  const [showAuth, setShowAuth] = useState(false);
  const [showContact, setShowContact] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [showForgotPwd, setShowForgotPwd] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, status: 'initializing' });
  const [errorMsg, setErrorMsg] = useState(null);

  const [showTextPaste, setShowTextPaste] = useState(false);
  const [pasteContent, setPasteContent] = useState('');

  const showError = (msg) => {
    setErrorMsg(msg);
    setView('home');
    setTimeout(() => setErrorMsg(null), 6000);
  };

  // ── Logged-in user (persisted) ──
  const [currentUser, setCurrentUser] = useState(() => {
    try { const u = localStorage.getItem('df_current_user'); return u ? JSON.parse(u) : null; }
    catch { return null; }
  });

  const handleLogin = (user) => setCurrentUser(user);
  const handleLogout = () => {
    localStorage.removeItem('df_current_user');
    setCurrentUser(null);
  };

  // ── Theme ──
  const [isDark, setIsDark] = useState(() => {
    const s = localStorage.getItem('df_theme'); return s ? s === 'dark' : true;
  });
  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', !isDark);
    localStorage.setItem('df_theme', isDark ? 'dark' : 'light');
  }, [isDark]);
  const toggleTheme = () => setIsDark(v => !v);
  
  const resetToHome = () => {
    setAnalysisResult(null);
    setSelectedModality(null);
    setView('home');
    setProgress({ percent: 0, status: 'initializing' });
    window.scrollTo(0, 0);
  };

  const pollJobStatus = async (jobId) => {
    try {
      const res = await axios.get(`${API_BASE}/job/${jobId}`);
      const job = res.data;
      setProgress({ percent: job.progress || 0, status: job.status });
      if (job.status === 'completed') { setAnalysisResult(job.result); setView('result'); }
      else if (job.status === 'failed') {
        const errDetail = job.error || 'Analysis failed. Please try a different file.';
        showError(`❌ Analysis Failed: ${errDetail}`);
      }
      else setTimeout(() => pollJobStatus(jobId), 1000);
    } catch (err) {
      const detail = err.response?.data?.message || err.response?.data?.detail || err.message;
      showError(`❌ Connection Error: ${detail}. Is the backend running on ${API_BASE}?`);
    }
  };

  const startAnalysis = async (inputData, modality) => {
    setSelectedModality(modality); setView('scanning'); setProgress({ percent: 0, status: 'uploading' });
    try {
      const formData = new FormData();
      let endpoint = '';
      if (modality === 'text') { endpoint = '/predict_text'; formData.append('current_text', inputData); }
      else {
        endpoint = modality === 'image' ? '/predict_image' : modality === 'audio' ? '/predict_audio' : '/predict';
        formData.append('file', inputData);
      }
      const response = await axios.post(`${API_BASE}${endpoint}`, formData);
      if (response.data.job_id) pollJobStatus(response.data.job_id);
      else { setAnalysisResult(response.data); setView('result'); }
    } catch (err) {
      const detail = err.response?.data?.message || err.response?.data?.detail || err.message;
      const statusCode = err.response?.status;
      if (statusCode === 413) showError(`❌ File Too Large: ${detail}`);
      else if (statusCode === 400) showError(`❌ Invalid File: ${detail}`);
      else showError(`❌ Upload Failed: ${detail}. Check that the backend is running.`);
    }
  };

  const handleUrlAnalysis = async (modality) => {
    const url = urlInputs[modality]?.trim();
    if (!url) { showError('⚠️ Please paste a URL first.'); return; }
    if (!url.startsWith('http://') && !url.startsWith('https://') && !url.startsWith('data:')) {
      showError('⚠️ URL must start with http://, https:// or data:');
      return;
    }
    setSelectedModality(modality); setView('scanning'); setProgress({ percent: 0, status: 'downloading_url' });
    try {
      const formData = new FormData(); formData.append('url', url); formData.append('modality', modality);
      const response = await axios.post(`${API_BASE}/predict_url`, formData);
      if (response.data.job_id) pollJobStatus(response.data.job_id);
      else { setAnalysisResult(response.data); setView('result'); }
    } catch (err) {
      const detail = err.response?.data?.message || err.response?.data?.detail || err.message;
      showError(`❌ URL Analysis Failed: ${detail}`);
    }
  };

  const ALLOWED_TYPES = {
    image: ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
    video: ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'],
    audio: ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac', 'audio/webm'],
  };
  const MAX_FILE_MB = { image: 10, video: 200, audio: 50 };

  const handleFileUpload = (e, modality) => {
    const f = e.target.files[0]; if (!f) return;
    const allowed = ALLOWED_TYPES[modality] || [], maxMB = MAX_FILE_MB[modality] || 100, sizeMB = f.size / (1024 * 1024);
    if (!allowed.includes(f.type)) {
      showError(`❌ Invalid file type for ${modality}. Allowed: ${allowed.map(t => t.split('/')[1].toUpperCase()).join(', ')}`);
      e.target.value = ''; return;
    }
    if (sizeMB > maxMB) {
      showError(`❌ File too large: ${sizeMB.toFixed(1)} MB. Maximum for ${modality} is ${maxMB} MB.`);
      e.target.value = ''; return;
    }
    startAnalysis(f, modality);
  };

  const handleDragOver = e => { e.preventDefault(); if (!isDragging) setIsDragging(true); };
  const handleDragLeave = e => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = e => {
    e.preventDefault(); setIsDragging(false);
    const f = e.dataTransfer.files[0]; if (!f) return;
    if (f.type.startsWith('image/')) handleFileUpload({ target: { files: [f] } }, 'image');
    else if (f.type.startsWith('video/')) handleFileUpload({ target: { files: [f] } }, 'video');
    else if (f.type.startsWith('audio/')) handleFileUpload({ target: { files: [f] } }, 'audio');
    else showError('❌ Unsupported file type. Drag an image, video, or audio file.');
  };

  const modalityColors = {
    image: { accent: '#7ec8a0', glow: 'rgba(126,200,160,0.3)', bg: '#7ec8a0', border: 'rgba(126,200,160,0.4)', inputBorder: 'rgba(126,200,160,0.3)' },
    video: { accent: '#6390ff', glow: 'rgba(99,144,255,0.3)',  bg: '#6390ff', border: 'rgba(99,144,255,0.4)',  inputBorder: 'rgba(99,144,255,0.3)'  },
    audio: { accent: '#ff9063', glow: 'rgba(255,144,99,0.3)',  bg: '#ff9063', border: 'rgba(255,144,99,0.4)',  inputBorder: 'rgba(255,144,99,0.3)'  },
    text:  { accent: '#c87eff', glow: 'rgba(200,126,255,0.3)', bg: '#c87eff', border: 'rgba(200,126,255,0.4)', inputBorder: 'rgba(200,126,255,0.3)' },
  };

  const ModalityCard = ({ modality, icon: Icon, title, description, uploadAccept, uploadId, captureBtnText, captureIcon: CaptureIcon }) => {
    const c = modalityColors[modality];
    return (
      <div id={`${modality}-card`}
        className="glass-morphism rounded-3xl p-8 flex flex-col items-center text-center border transition-all hover:scale-105 shadow-xl relative overflow-hidden group"
        style={{ borderColor: c.border, boxShadow: `0 4px 30px ${c.glow}00` }}
        onMouseEnter={e => e.currentTarget.style.boxShadow = `0 4px 30px ${c.glow}`}
        onMouseLeave={e => e.currentTarget.style.boxShadow = `0 4px 30px ${c.glow}00`}>
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
          style={{ backgroundImage: `linear-gradient(to bottom, ${c.accent}18, transparent)` }} />
        <Icon size={48} className="mb-4" style={{ color: c.accent }} />
        <h2 className="text-2xl font-bold mb-2">{title}</h2>
        <p className="text-sm text-gray-400 mb-6 flex-1">{description}</p>
        <div className="w-full space-y-3 z-10 relative">
          {modality !== 'text' ? (
            <>
              <button onClick={() => document.getElementById(uploadId).click()}
                className="w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all hover:opacity-90"
                style={{ background: c.bg, color: modality === 'image' ? '#000' : '#fff' }}>
                <UploadCloud size={18} /> Upload {title}
              </button>
              <input id={uploadId} type="file" accept={uploadAccept} className="hidden" onChange={e => handleFileUpload(e, modality)} />
              <button onClick={() => { setSelectedModality(modality); setIsCaptureViewOpen(true); }}
                className="w-full py-3 rounded-xl font-bold border flex items-center justify-center gap-2 transition-all hover:opacity-90"
                style={{ borderColor: c.accent, color: c.accent, background: `${c.accent}12` }}>
                {CaptureIcon && <CaptureIcon size={18} />} {captureBtnText}
              </button>
              <div className="flex gap-2">
                <input id={`${modality}-url-input`} type="url" placeholder={`Paste ${title.toLowerCase()} URL…`}
                  value={urlInputs[modality] || ''}
                  onChange={e => setUrlInputs(p => ({ ...p, [modality]: e.target.value }))}
                  onKeyDown={e => e.key === 'Enter' && handleUrlAnalysis(modality)}
                  className="flex-1 px-3 py-2 rounded-xl text-sm placeholder-gray-500 focus:outline-none transition-colors"
                  style={{ background: isDark ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.06)', border: `1px solid ${c.inputBorder}`, color: isDark ? '#fff' : '#0f172a' }} />
                <button onClick={() => handleUrlAnalysis(modality)} title="Analyze URL"
                  className="px-3 py-2 rounded-xl transition-all hover:scale-110"
                  style={{ background: `${c.accent}22`, border: `1px solid ${c.accent}66`, color: c.accent }}>
                  <ArrowRight size={16} />
                </button>
              </div>
            </>
          ) : (
            <>
              <button id="paste-text-btn" onClick={() => { console.log("Opening text paste modal..."); setShowTextPaste(true); }}
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

  const navTextColor = isDark ? '#fff' : '#1e293b';

  return (
    <div className="min-h-screen relative overflow-hidden"
      style={{ backgroundColor: isDark ? '#000' : '#f0f4f8', color: isDark ? '#fff' : '#0f172a' }}
      onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
      <StarBackground />
      <AIAssistant isDark={isDark} />

      {/* ── Error Toast Notification ── */}
      {errorMsg && (
        <div className="fixed top-6 left-1/2 z-[500] fade-up" style={{ transform: 'translateX(-50%)', maxWidth: '520px', width: '90vw' }}>
          <div className="flex items-start gap-3 px-5 py-4 rounded-2xl shadow-2xl border text-sm font-semibold"
            style={{ background: 'rgba(20,8,8,0.97)', borderColor: 'rgba(239,68,68,0.5)', color: '#fca5a5', backdropFilter: 'blur(16px)' }}>
            <span style={{ fontSize: 18, flexShrink: 0 }}>⚠️</span>
            <span className="flex-1 leading-relaxed">{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="hover:opacity-100 transition-opacity font-black text-lg leading-none"
              style={{ color: '#fca5a5', flexShrink: 0, opacity: 0.7 }}>✕</button>
          </div>
        </div>
      )}

      {/* ── Modals ── */}
      {showAuth && (
        <AuthModal isDark={isDark} onClose={() => setShowAuth(false)}
          onForgotPassword={() => { setShowAuth(false); setShowForgotPwd(true); }}
          onLogin={handleLogin} />
      )}
      {showContact && <ContactModal isDark={isDark} onClose={() => setShowContact(false)} />}
      {showAbout   && <AboutModal   isDark={isDark} onClose={() => setShowAbout(false)}   />}
      {showForgotPwd && (
        <ForgotPasswordModal isDark={isDark}
          onClose={() => { setShowForgotPwd(false); setShowAuth(true); }} />
      )}

      {showTextPaste && (
        <div className="fixed inset-0 z-[400] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="bg-[#0f172a] border border-[#c87eff]/40 w-full max-w-2xl rounded-3xl p-8 shadow-2xl fade-up">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'rgba(200,126,255,0.15)', border: '1px solid rgba(200,126,255,0.3)' }}>
                  <Type size={20} style={{ color: '#c87eff' }} />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Semantic Text Analysis</h2>
                  <p className="text-xs text-gray-400">Detect AI-generated patterns and LLM signatures.</p>
                </div>
              </div>
              <button onClick={() => setShowTextPaste(false)} className="text-gray-500 hover:text-white transition-colors">
                <X size={20} />
              </button>
            </div>
            <textarea
              value={pasteContent}
              onChange={e => setPasteContent(e.target.value)}
              placeholder="Paste the text you want to analyze here... (e.g. news articles, emails, or chat logs)"
              className="w-full h-64 bg-black/40 border border-white/10 rounded-2xl p-4 text-sm text-gray-200 focus:outline-none focus:border-[#c87eff]/60 transition-colors resize-none mb-6"
            />
            <div className="flex gap-4">
              <button onClick={() => setShowTextPaste(false)} className="flex-1 py-3 rounded-xl font-bold text-gray-400 hover:text-white transition-colors">
                Cancel
              </button>
              <button 
                disabled={!pasteContent.trim()}
                onClick={() => {
                  if (pasteContent.trim()) {
                    startAnalysis(pasteContent, 'text');
                    setShowTextPaste(false);
                    setPasteContent('');
                  }
                }}
                className={`flex-1 py-3 rounded-xl font-bold text-white transition-all ${!pasteContent.trim() ? 'opacity-30 grayscale cursor-not-allowed' : 'hover:scale-105 active:scale-95'}`}
                style={{ background: 'linear-gradient(135deg,#c87eff,#9d58cc)' }}>
                Analyze Semantic Patterns
              </button>
            </div>
          </div>
        </div>
      )}

      {isDragging && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm border-4 border-dashed border-[#7ec8a0]">
          <h2 className="text-4xl font-bold text-[#7ec8a0] animate-pulse">Drop File to Analyze</h2>
        </div>
      )}

      {/* ─── NAVBAR ─── */}
      <nav className="relative z-50 flex items-center justify-between px-10 py-5">
        <div className="flex items-center gap-4 cursor-pointer" onClick={resetToHome}>
          <div className="w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold text-black"
            style={{ backgroundColor: '#7ec8a0', boxShadow: '0 0 20px rgba(126,200,160,0.6)' }}>DF</div>
          <span className="text-3xl font-bold tracking-wide"
            style={{ color: '#7ec8a0', textShadow: '0 0 10px rgba(126,200,160,0.4)' }}>DeepFake</span>
        </div>

        <div className="hidden md:flex items-center gap-4">
          <button onClick={() => setView('home')}
            className="nav-link font-semibold text-sm tracking-wide transition-colors hover:text-[#7ec8a0]"
            style={{ color: navTextColor }}>Home</button>

          {/* About — now functional */}
          <button id="about-nav-btn" onClick={() => setShowAbout(true)}
            className="nav-link font-semibold text-sm tracking-wide transition-colors hover:text-[#7ec8a0]"
            style={{ color: navTextColor }}>About</button>

          <button id="analytics-nav-btn" onClick={() => setView('analytics')}
            className="nav-link font-semibold text-sm tracking-wide transition-colors hover:text-[#6390ff]"
            style={{ color: navTextColor }}>Analytics</button>

          {/* Theme Toggle */}
          <ThemeToggle isDark={isDark} onToggle={toggleTheme} />

          {/* Auth section — shows user avatar if logged in, else Login button */}
          {currentUser ? (
            <UserNav user={currentUser} onLogout={handleLogout} isDark={isDark} />
          ) : (
            <button id="login-nav-btn" onClick={() => setShowAuth(true)}
              className="px-5 py-2 rounded-full text-sm font-bold border transition-all hover:scale-105"
              style={{ borderColor: '#7ec8a0', color: '#7ec8a0', background: 'rgba(126,200,160,0.08)' }}>
              Login / Sign Up
            </button>
          )}

          {/* Contact Us */}
          <button id="contact-nav-btn" onClick={() => setShowContact(true)}
            className="px-5 py-2 rounded-full text-sm font-bold transition-all hover:scale-105"
            style={{ background: 'linear-gradient(135deg,#6390ff,#4a72d1)', color: '#fff' }}>
            Contact Us
          </button>
          
          {view !== 'home' && (
            <button onClick={resetToHome} 
              className="px-5 py-2 rounded-full text-sm font-black border border-pistachio text-pistachio hover:bg-pistachio/10 transition-all">
              Back to Home
            </button>
          )}
        </div>
      </nav>

      {/* ─── MAIN CONTENT ─── */}
      <main className="relative z-10 w-full min-h-[85vh] flex items-center justify-center px-10">
        {view === 'home' && (
          <div className="w-full max-w-[90vw] flex flex-col items-center mt-10 space-y-12">
            <div className="text-center space-y-4">
              {currentUser && (
                <p className="text-sm font-semibold" style={{ color: '#7ec8a0' }}>
                  👋 Welcome, {currentUser.name || currentUser.email}!
                </p>
              )}
              <h1 className="text-5xl lg:text-7xl font-bold tracking-tight glow-text leading-tight">
                Deepfake Intelligence
              </h1>
              <p className="text-xl max-w-2xl mx-auto" style={{ color: isDark ? '#d1d5db' : '#475569' }}>
                Select a modality to begin forensic analysis. Upload, capture live, or paste a URL.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-8 w-full pb-20">
              <ModalityCard modality="image" icon={UploadCloud} title="Image" uploadAccept="image/*" uploadId="image-upload"
                captureBtnText="Live Snapshot" captureIcon={Camera}
                description="High-velocity EfficientNet-V2 with spectral DCT boosters and forensic ELA heatmaps." />
              <ModalityCard modality="video" icon={Play} title="Video" uploadAccept="video/*" uploadId="video-upload"
                captureBtnText="Record Video" captureIcon={Camera}
                description="Multimodal fusion (visual + audio) with ResNet50-LSTM temporal synchronization." />
              <ModalityCard modality="audio" icon={Mic} title="Audio" uploadAccept="audio/*" uploadId="audio-upload"
                captureBtnText="Record Audio" captureIcon={Mic}
                description="Wav2Vec 2.0 Feature correlation engine to distinguish human from cloned AI speech." />
              <ModalityCard modality="text" icon={Type} title="Text"
                description="Advanced semantic analysis with RoBERTa transformers to identify AI writing patterns." />
            </div>
            <LandingContent isDark={isDark} />
          </div>
        )}
        {view === 'scanning' && selectedModality && <ScannerView modality={selectedModality} progress={progress} onBack={resetToHome} />}
        {isCaptureViewOpen && (
          <CaptureView modality={selectedModality}
            onCapture={file => { setIsCaptureViewOpen(false); startAnalysis(file, selectedModality); }}
            onClose={() => setIsCaptureViewOpen(false)} />
        )}
        {view === 'result' && analysisResult && (
          <div className="w-full h-[85vh] overflow-y-auto custom-scrollbar relative">
            <ResultDashboard 
              result={analysisResult} 
              onReset={resetToHome} 
            />
          </div>
        )}
        {view === 'analytics' && <AnalyticsDashboard onBack={() => setView('home')} isDark={isDark} />}
      </main>
    </div>
  );
}
