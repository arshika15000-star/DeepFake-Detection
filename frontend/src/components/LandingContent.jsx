import React, { useState } from 'react';
import { 
  UploadCloud, Brain, FileCheck, Share2, 
  BookOpen, Video, Shield, PhoneOff, 
  ChevronDown, Hexagon, Fingerprint, Eye, Globe
} from 'lucide-react';

export default function LandingContent({ isDark, onStart }) {
  const [openFaq, setOpenFaq] = useState(null);

  const bg = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)';
  const borderCol = isDark ? 'rgba(126,200,160,0.2)' : 'rgba(126,200,160,0.3)';
  const textMain = isDark ? '#fff' : '#0f172a';
  const textSub = isDark ? '#94a3b8' : '#64748b';
  const cardBg = isDark ? 'rgba(15,15,25,0.6)' : '#ffffff';

  const faqs = [
    {
      q: "What is deepfake detection?",
      a: "Deepfake detection is the process of identifying AI-generated or digitally manipulated content—like fake images, videos, or voices—that are made to look and sound real. Our system analyzes signs of tampering, such as pixel inconsistencies, unnatural movements, or audio mismatches, to ensure media authenticity."
    },
    {
      q: "How does our multimodal detection work?",
      a: "Our engine uses specialized Deep Learning models for each modality. EfficientNet and XceptionNet scan for image artifacts; rPPG and BlazeFace analyze video temporal coherence; Wav2Vec 2.0 flags synthetic audio features; and BERT analyzes textual anomalies. The results are fused into a comprehensive authenticity score."
    },
    {
      q: "Are the results explainable?",
      a: "Yes! Unlike black-box models, our Explainable AI (XAI) approach uses GradCAM heatmaps to visually highlight the exact regions in an image or video that triggered a 'fake' classification. For audio, we display mel-spectrogram anomalies, giving you clear reasoning."
    },
    {
      q: "How fast is the analysis?",
      a: "Most analyses complete within seconds. Image and audio files are analyzed almost instantly, while video processing time scales efficiently based on the length and resolution of the clip."
    },
    {
      q: "Is my data secure?",
      a: "Absolutely. We do not store your files. All media is analyzed in memory and immediately discarded. Your privacy is our top priority."
    }
  ];

  const steps = [
    { icon: UploadCloud, title: "Step 1: Upload Your File", desc: "Choose the file you want to check. We support images, videos, audio, and text via direct upload, live capture, or URL." },
    { icon: Brain, title: "Step 2: Multimodal AI Analysis", desc: "Our neural networks start scanning for deepfake manipulation using state-of-the-art forensic algorithms." },
    { icon: FileCheck, title: "Step 3: View Explainable Results", desc: "Get a clear authenticity score, along with GradCAM heatmaps and detailed feature insights that explain exactly why a file was flagged." },
    { icon: Share2, title: "Step 4: Secure & Share", desc: "Download your detailed PDF forensic report or securely share the results with your team." }
  ];

  const useCases = [
    { icon: BookOpen, title: "Educators & Researchers", desc: "Analyze historical or viral content to teach students digital literacy and the science behind synthetic media." },
    { icon: Video, title: "Journalists & Newsrooms", desc: "Verify breaking news clips and user-generated content before publishing to combat misinformation." },
    { icon: Shield, title: "Creators & Public Figures", desc: "Protect your digital identity by monitoring if your face or voice is being synthetically replicated." },
    { icon: PhoneOff, title: "Families & Individuals", desc: "Safeguard against voice-cloning scams and ensure the authenticity of suspicious messages or calls." }
  ];

  return (
    <div className="w-full flex flex-col items-center mt-12 space-y-24 pb-16 fade-up">

      {/* ─── ABOUT THIS PROJECT (Overview) ─── */}
      <section className="w-full max-w-5xl text-center px-4 space-y-6">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold mb-4"
          style={{ background: 'rgba(126,200,160,0.15)', color: '#7ec8a0', border: '1px solid rgba(126,200,160,0.3)' }}>
          <Fingerprint size={16} /> Explainable Multimodal Architecture
        </div>
        <h2 className="text-4xl md:text-5xl font-black tracking-tight" style={{ color: textMain }}>
          Unmasking the Illusion
        </h2>
        <p className="text-lg md:text-xl max-w-3xl mx-auto leading-relaxed" style={{ color: textSub }}>
          This project is an advanced, real-time forensic platform built to combat the rising threat of artificial media. 
          By seamlessly combining <strong>Computer Vision, NLP, and Audio processing</strong>, we provide a unified defense against deepfakes. 
          Our Explainable AI (XAI) doesn't just give you a score—it shows you <i>exactly</i> where the manipulation lies.
        </p>
      </section>

      {/* ─── HOW IT WORKS (Steps) ─── */}
      <section className="w-full max-w-6xl px-4">
        <h3 className="text-3xl font-bold text-center mb-12" style={{ color: textMain }}>
          Simple, Fast, and Transparent
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((s, i) => (
            <div key={i} className="relative p-6 rounded-3xl border transition-all hover:-translate-y-2 group"
              style={{ background: bg, borderColor: borderCol, isolation: 'isolate' }}>
              <div className="absolute top-0 right-0 p-4 opacity-10 text-6xl font-black" style={{ color: '#7ec8a0' }}>
                0{i + 1}
              </div>
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-6"
                style={{ background: 'linear-gradient(135deg,#7ec8a0,#5aaa80)', boxShadow: '0 8px 20px rgba(126,200,160,0.4)' }}>
                <s.icon size={28} color="#000" />
              </div>
              <h4 className="text-xl font-bold mb-3 z-10 relative" style={{ color: textMain }}>{s.title}</h4>
              <p className="text-sm leading-relaxed z-10 relative" style={{ color: textSub }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ─── USE CASES ─── */}
      <section className="w-full max-w-6xl px-4">
        <div className="flex flex-col items-center mb-12 text-center">
          <h3 className="text-3xl font-bold mb-4" style={{ color: textMain }}>Who Uses Our Platform?</h3>
          <p className="max-w-2xl text-base" style={{ color: textSub }}>
            From news desks to living rooms, our multimodal AI empowers everyone to verify digital realities.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {useCases.map((uc, i) => (
            <div key={i} className="flex gap-6 p-8 rounded-3xl border transition-all hover:scale-[1.02]"
              style={{ background: cardBg, borderColor: borderCol, boxShadow: isDark ? '0 10px 40px rgba(0,0,0,0.5)' : '0 10px 40px rgba(0,0,0,0.05)' }}>
              <div className="flex-shrink-0 w-16 h-16 rounded-full flex items-center justify-center"
                style={{ background: 'rgba(126,200,160,0.1)', color: '#7ec8a0' }}>
                <uc.icon size={32} />
              </div>
              <div>
                <h4 className="text-xl font-bold mb-2" style={{ color: textMain }}>{uc.title}</h4>
                <p className="text-sm leading-relaxed" style={{ color: textSub }}>{uc.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── STATS & FEATURES BANNER ─── */}
      <section className="w-full max-w-6xl px-4">
        <div className="rounded-3xl p-10 md:p-16 flex flex-col md:flex-row items-center justify-between gap-10"
          style={{ background: 'linear-gradient(135deg, rgba(126,200,160,0.1) 0%, rgba(99,144,255,0.05) 100%)', border: `1px solid ${borderCol}` }}>
          <div className="max-w-xl">
            <h3 className="text-3xl font-bold mb-4" style={{ color: textMain }}>Comprehensive Security</h3>
            <p className="text-base mb-6 leading-relaxed" style={{ color: textSub }}>
              This platform uses an ensemble approach. By integrating <strong>EfficientNet, XceptionNet, Wav2Vec, and BERT</strong>, we cover all angles of generation artifacts. 
              The system cross-references visual discrepancies, audio spectrum glitches, and linguistic unnaturalness.
            </p>
            <ul className="space-y-3">
              {['Highest Accuracy Detection', 'Zero Data Retention', 'Actionable Forensic Heatmaps', 'Free Academic Access'].map((li, i) => (
                <li key={i} className="flex items-center gap-3 text-sm font-semibold" style={{ color: textMain }}>
                  <span className="w-6 h-6 rounded-full flex items-center justify-center" style={{ background: '#7ec8a0', color: '#000' }}>✓</span>
                  {li}
                </li>
              ))}
            </ul>
          </div>
          <div className="grid grid-cols-2 gap-4 flex-shrink-0 w-full md:w-auto">
            <div className="p-6 rounded-2xl text-center" style={{ background: cardBg, border: `1px solid ${borderCol}` }}>
              <div className="text-4xl font-black mb-1" style={{ color: '#7ec8a0' }}>100%</div>
              <div className="text-xs font-bold uppercase tracking-wider" style={{ color: textSub }}>Private Analysis</div>
            </div>
            <div className="p-6 rounded-2xl text-center" style={{ background: cardBg, border: `1px solid ${borderCol}` }}>
              <div className="text-4xl font-black mb-1" style={{ color: '#6390ff' }}>&lt;5s</div>
              <div className="text-xs font-bold uppercase tracking-wider" style={{ color: textSub }}>Processing Time</div>
            </div>
            <div className="p-6 rounded-2xl text-center col-span-2" style={{ background: cardBg, border: `1px solid ${borderCol}` }}>
              <div className="text-3xl font-black mb-1" style={{ color: '#c87eff' }}>4</div>
              <div className="text-xs font-bold uppercase tracking-wider" style={{ color: textSub }}>Modalities Supported</div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── FAQs ─── */}
      <section className="w-full max-w-4xl px-4">
        <h3 className="text-3xl font-bold text-center mb-10" style={{ color: textMain }}>Frequently Asked Questions</h3>
        <div className="space-y-4">
          {faqs.map((faq, index) => {
            const isOpen = openFaq === index;
            return (
              <div key={index} 
                className="border rounded-2xl overflow-hidden transition-all duration-300"
                style={{ borderColor: borderCol, background: isOpen ? bg : 'transparent' }}>
                <button 
                  className="w-full text-left px-6 py-5 flex items-center justify-between focus:outline-none"
                  onClick={() => setOpenFaq(isOpen ? null : index)}>
                  <h4 className="font-bold text-lg" style={{ color: textMain }}>{faq.q}</h4>
                  <ChevronDown className={`transform transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} style={{ color: '#7ec8a0' }} />
                </button>
                <div 
                  className="px-6 transition-all duration-300 ease-in-out"
                  style={{ 
                    maxHeight: isOpen ? '500px' : '0px', 
                    opacity: isOpen ? 1 : 0,
                    paddingBottom: isOpen ? '1.25rem' : '0px'
                  }}>
                  <p className="text-sm leading-relaxed" style={{ color: textSub }}>{faq.a}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ─── FOOTER ─── */}
      <footer className="w-full max-w-7xl px-4 mt-20 pt-10 border-t flex flex-col items-center text-center" style={{ borderColor: borderCol }}>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-black" style={{ background: '#7ec8a0' }}>DF</div>
          <span className="text-2xl font-bold" style={{ color: '#7ec8a0' }}>Deepfake Detection</span>
        </div>
        <p className="max-w-xl text-sm mb-8" style={{ color: textSub }}>
          An open academic platform providing cutting-edge, multimodal, and highly explainable AI tools to detect and analyze synthetic media in real time.
        </p>
        <div className="flex flex-wrap justify-center gap-6 text-sm font-semibold mb-10" style={{ color: textMain }}>
          <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth'})} className="hover:text-[#7ec8a0] transition-colors">Start Analysis</button>
          <button className="hover:text-[#7ec8a0] transition-colors">Privacy Policy</button>
          <button className="hover:text-[#7ec8a0] transition-colors">Contact Us</button>
        </div>
        <p className="text-xs" style={{ color: textSub }}>© {new Date().getFullYear()} Deepfake Detection AI. All rights reserved.</p>
      </footer>

    </div>
  );
}
