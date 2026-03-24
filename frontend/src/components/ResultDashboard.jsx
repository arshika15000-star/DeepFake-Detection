import React from 'react';
import { motion } from 'framer-motion';
import {
    ShieldCheck, AlertTriangle, Activity, Download, RefreshCcw, Layers, Eye, FileText, Workflow, Fingerprint, Mic, Type, PlaySquare, Image as ImageIcon
} from 'lucide-react';
import FeedbackPanel from './FeedbackPanel';

export default function ResultDashboard({ result, onReset }) {
    const isFake = result.prediction === 'FAKE';
    const confidence = Math.round(result.confidence * 100);

    // Infer media type and extract relevant forensics
    const isVideo = !!result.frames_processed;
    const isAudio = result.forensics?.vocal_jitter !== undefined;
    const isText = result.forensics?.complexity_index !== undefined;
    const isImage = !isVideo && !isAudio && !isText;
    const mediaType = isVideo ? 'video' : isAudio ? 'audio' : isText ? 'text' : 'image';

    const forensics = result.forensics || {};
    const findings = forensics.findings || result.metadata?.findings || forensics.metadata?.findings || ["No specific anomalies detected."];

    // Helper: format percent
    const pct = (val) => Math.round(val * 100);

    return (
        <div className="min-h-screen pt-20 pb-12 px-6 lg:px-12 max-w-7xl mx-auto space-y-8 fade-up" style={{ color: 'var(--text-main)' }}>
            {/* Header Section */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 glass-morphism p-8 rounded-[2rem] shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -z-10 translate-x-1/2 -translate-y-1/2" />
                <div className="flex items-center gap-6 z-10">
                    <div className={`p-6 rounded-3xl border shadow-xl ${isFake ? 'bg-danger/10 text-danger border-danger/20' : 'bg-success/10 text-success border-success/20'}`}>
                        {isFake ? <AlertTriangle size={56} /> : <ShieldCheck size={56} />}
                    </div>
                    <div>
                        <div className="flex items-center gap-3 mb-1">
                            <span className="text-[10px] font-black uppercase tracking-[0.4em]" style={{ color: 'var(--text-dim)', opacity: 0.8 }}>
                                {mediaType} Verdict
                            </span>
                            <div className={`h-1.5 w-1.5 rounded-full animate-ping ${isFake ? 'bg-danger' : 'bg-success'}`} />
                        </div>
                        <h1 className={`text-6xl max-md:text-5xl font-black tracking-tighter ${isFake ? 'text-danger' : 'text-success'}`}>
                            {result.prediction}
                        </h1>
                        {isFake && forensics.source_attribution && (
                            <div className="mt-2 text-xs font-bold px-3 py-1 rounded-full bg-danger/10 text-danger border border-danger/20 inline-flex items-center gap-2">
                                <Fingerprint size={12} />
                                Predicted Generator: {forensics.source_attribution.most_likely} ({pct(forensics.source_attribution.confidence)}%)
                            </div>
                        )}
                    </div>
                </div>

                <div className="flex gap-4 z-10 w-full md:w-auto">
                    <button
                        className="flex-1 md:flex-none flex items-center justify-center gap-3 glass-morphism px-8 py-4 rounded-2xl hover:bg-white/5 transition-all font-bold text-sm tracking-widest uppercase border"
                        style={{ color: 'var(--text-main)', borderColor: 'var(--border-subtle)' }}
                        onClick={() => window.print()}
                    >
                        <Download size={18} /> Export
                    </button>
                    <button
                        className="flex-1 md:flex-none btn-glow flex items-center justify-center gap-3 px-8 py-4 rounded-2xl shadow-lg hover:scale-105 transition-all font-black text-sm tracking-widest uppercase border border-transparent"
                        onClick={onReset}
                    >
                        <RefreshCcw size={18} /> New Analysis
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Confidence Meter */}
                <div className="lg:col-span-1 glass-morphism p-8 rounded-[2rem] flex flex-col items-center justify-center text-center space-y-8 relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                    <h3 className="text-[10px] font-black uppercase tracking-[0.5em] z-10" style={{ color: 'var(--text-dim)' }}>Neural Confidence</h3>

                    <div className="relative w-64 h-64 flex items-center justify-center z-10">
                        <svg className="w-full h-full -rotate-90 filter drop-shadow-xl">
                            <circle
                                cx="128" cy="128" r="110"
                                className="fill-none"
                                stroke="var(--border-subtle)"
                                strokeWidth="18"
                            />
                            <motion.circle
                                cx="128" cy="128" r="110"
                                className={`fill-none ${isFake ? 'stroke-danger' : 'stroke-primary'}`}
                                strokeWidth="18"
                                strokeDasharray="691.15"
                                initial={{ strokeDashoffset: 691.15 }}
                                animate={{ strokeDashoffset: 691.15 - (691.15 * confidence / 100) }}
                                transition={{ duration: 1.5, ease: "easeOut" }}
                                strokeLinecap="round"
                            />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-6xl font-black tracking-tighter" style={{ color: 'var(--text-main)' }}>{confidence}%</span>
                            <span className="text-[10px] font-bold uppercase tracking-widest mt-1" style={{ color: 'var(--text-dim)' }}>Certainty Index</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 w-full z-10">
                        <div className="rounded-2xl p-4 border bg-black/20 backdrop-blur-md" style={{ borderColor: 'var(--border-subtle)' }}>
                            <div className="text-[10px] font-bold uppercase mb-1" style={{ color: 'var(--text-dim)' }}>Authenticity</div>
                            <div className="text-2xl font-black text-success tracking-tighter">{100 - confidence}%</div>
                        </div>
                        <div className="rounded-2xl p-4 border bg-black/20 backdrop-blur-md" style={{ borderColor: 'var(--border-subtle)' }}>
                            <div className="text-[10px] font-bold uppercase mb-1" style={{ color: 'var(--text-dim)' }}>Manipulation</div>
                            <div className="text-2xl font-black text-danger tracking-tighter">{confidence}%</div>
                        </div>
                    </div>
                </div>

                {/* Explainable AI Metrics */}
                <div className="lg:col-span-2 glass-morphism p-8 rounded-[2rem] space-y-8 flex flex-col">
                    <div className="flex justify-between items-center border-b pb-4" style={{ borderColor: 'var(--border-subtle)' }}>
                        <h3 className="text-xl font-black tracking-wide flex items-center gap-3" style={{ color: 'var(--text-main)' }}>
                            <Workflow size={24} style={{ color: 'var(--primary)' }} /> XAI ANALYSIS CLUSTERS
                        </h3>
                        <span className="text-[10px] font-bold uppercase border px-3 py-1 rounded-full bg-primary/10" style={{ color: 'var(--primary)', borderColor: 'var(--primary)' }}>
                            {mediaType.toUpperCase()} EXPLAINER
                        </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1">
                        
                        {/* Image Modality */}
                        {(isImage || (isVideo && forensics.heatmap)) && (
                            <div className="rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <Eye size={20} style={{ color: 'var(--primary)' }} />
                                        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Attention Map (Grad-CAM)</span>
                                    </div>
                                    <ImageIcon size={16} style={{ color: 'var(--text-dim)', opacity: 0.5 }} />
                                </div>
                                <div className="h-40 rounded-xl overflow-hidden border relative flex items-center justify-center transition-all bg-black/20" style={{ borderColor: 'var(--border-subtle)' }}>
                                    {forensics.heatmap ? (
                                        <img src={`data:image/jpeg;base64,${forensics.heatmap}`} className="w-full h-full object-cover opacity-90 hover:opacity-100 transition-opacity hover:scale-105 duration-500" alt="Heatmap" />
                                    ) : (
                                        <div className="text-[10px] uppercase font-bold tracking-widest" style={{ color: 'var(--text-dim)' }}>Heatmap Unavailable</div>
                                    )}
                                </div>
                            </div>
                        )}
                        {isImage && forensics.ela && (
                            <div className="rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <Layers size={20} style={{ color: 'var(--secondary)' }} />
                                        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Error Level Analysis</span>
                                    </div>
                                    <Activity size={16} style={{ color: 'var(--text-dim)', opacity: 0.5 }} />
                                </div>
                                <div className="h-40 rounded-xl overflow-hidden border relative flex items-center justify-center transition-all bg-black/20" style={{ borderColor: 'var(--border-subtle)' }}>
                                    <img src={`data:image/jpeg;base64,${forensics.ela}`} className="w-full h-full object-cover opacity-90 filter contrast-125 brightness-110" alt="ELA" />
                                </div>
                            </div>
                        )}

                        {/* Video Modality */}
                        {isVideo && (
                            <div className="rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <PlaySquare size={20} style={{ color: 'var(--secondary)' }} />
                                        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Deep Neural Metrics</span>
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    {forensics.neural_metrics && Object.entries(forensics.neural_metrics).map(([key, val], i) => (
                                        <div key={key} className="space-y-1">
                                            <div className="flex justify-between text-[8px] font-black px-1" style={{ color: 'var(--text-dim)' }}>
                                                <span>{key.replace(/_/g, ' ').toUpperCase()}</span>
                                                <span>{pct(val)}%</span>
                                            </div>
                                            <div className="h-2 w-full rounded-full overflow-hidden bg-black/30" style={{ border: '1px solid var(--border-subtle)' }}>
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${pct(val)}%` }}
                                                    transition={{ duration: 1, delay: i * 0.2 }}
                                                    className="h-full shadow-lg"
                                                    style={{ background: 'linear-gradient(90deg, var(--primary), var(--secondary))' }}
                                                />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Video rPPG Pulse */}
                        {isVideo && result.pulse_data && (
                             <div className="rounded-2xl p-6 border group transition-all flex flex-col" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <Activity size={20} style={{ color: 'var(--danger)' }} />
                                        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>rPPG Neural Pulse</span>
                                    </div>
                                </div>
                                <div className="flex-1 flex items-end justify-between gap-1 h-32 pt-4">
                                    {result.pulse_data.slice(0, 30).map((p, i) => (
                                        <motion.div
                                            key={i}
                                            className="w-full rounded-t-sm"
                                            style={{ background: 'var(--primary)', opacity: 0.8 }}
                                            initial={{ height: 0 }}
                                            animate={{ height: `${(p / 100) * 100}%` }}
                                            transition={{ duration: 0.5, delay: i * 0.02 }}
                                        />
                                    ))}
                                </div>
                             </div>
                        )}

                        {/* Audio Modality */}
                        {isAudio && (
                            <div className="col-span-1 md:col-span-2 rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <Mic size={20} style={{ color: 'var(--primary)' }} />
                                        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Acoustic Fingerprint Analysis</span>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-black/20 p-4 rounded-xl border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase mb-2" style={{ color: 'var(--text-dim)' }}>Vocal Jitter Factor</div>
                                        <div className="text-3xl font-black" style={{ color: 'var(--primary)' }}>{pct(forensics.vocal_jitter || 0)}%</div>
                                        <div className="text-xs mt-1" style={{ color: 'var(--text-dim)' }}>Micro-fluctuations in pitch</div>
                                    </div>
                                    <div className="bg-black/20 p-4 rounded-xl border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase mb-2" style={{ color: 'var(--text-dim)' }}>Spectral Floor</div>
                                        <div className="text-sm font-bold mt-2 uppercase" style={{ color: 'var(--secondary)' }}>{forensics.spectral_floor || "Unknown"}</div>
                                        <div className="text-[10px] mt-2" style={{ color: 'var(--text-dim)' }}>Background noise profile</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Text Modality */}
                        {isText && (
                            <div className="col-span-1 md:col-span-2 rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <Type size={20} style={{ color: 'var(--primary)' }} />
                                        <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Linguistic Structure Analysis</span>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-black/20 p-4 rounded-xl border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase mb-2" style={{ color: 'var(--text-dim)' }}>Complexity Index</div>
                                        <div className="text-3xl font-black" style={{ color: 'var(--primary)' }}>{(forensics.complexity_index || 0).toFixed(2)}</div>
                                        <div className="text-[10px] mt-1" style={{ color: 'var(--text-dim)' }}>Vocabulary richness & sentence depth</div>
                                    </div>
                                    <div className="bg-black/20 p-4 rounded-xl border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase mb-2" style={{ color: 'var(--text-dim)' }}>Structural Rigidity</div>
                                        <div className={`text-sm font-bold mt-2 uppercase ${forensics.is_structured ? 'text-danger' : 'text-success'}`}>
                                            {forensics.is_structured ? "High (Machine-like)" : "Normal (Human-like)"}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Detailed Findings List */}
                    <div className="space-y-4 p-6 rounded-2xl border flex-none" style={{ background: 'rgba(126,200,160,0.05)', borderColor: 'var(--border-subtle)' }}>
                        <div className="flex items-center gap-3 text-xs font-bold uppercase tracking-widest mb-2" style={{ color: 'var(--text-dim)' }}>
                            <FileText size={16} style={{ color: 'var(--primary)' }} /> Forensic Execution Logs
                        </div>
                        <div className="space-y-3 font-mono text-xs overflow-y-auto max-h-32 pr-2 custom-scrollbar">
                            {findings.length > 0 ? findings.map((log, i) => (
                                <div key={i} className="flex gap-4 group items-start">
                                    <span style={{ color: 'var(--primary)' }}>[{String(i + 1).padStart(2, '0')}]</span>
                                    <span style={{ color: 'var(--text-dim)' }} className="group-hover:text-primary transition-colors leading-relaxed">{log}</span>
                                </div>
                            )) : (
                                <div className="flex gap-4 group items-start">
                                    <span style={{ color: 'var(--primary)' }}>[01]</span>
                                    <span style={{ color: 'var(--text-dim)' }}>No deterministic sub-artifacts registered for this sequence.</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Feedback Section */}
            <FeedbackPanel result={result} mediaType={mediaType} />
        </div>
    );
}
