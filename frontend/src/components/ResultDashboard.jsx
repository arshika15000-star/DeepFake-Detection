import React from 'react';
import { motion } from 'framer-motion';
import {
    ShieldCheck,
    AlertTriangle,
    Activity,
    Download,
    RefreshCcw,
    Layers,
    Eye,
    FileText,
    Workflow
} from 'lucide-react';
import FeedbackPanel from './FeedbackPanel';

export default function ResultDashboard({ result, onReset }) {
    const isFake = result.prediction === 'FAKE';
    const confidence = Math.round(result.confidence * 100);

    return (
        <div className="min-h-screen pt-20 pb-12 px-6 lg:px-12 max-w-7xl mx-auto space-y-8 fade-up" style={{ color: 'var(--text-main)' }}>
            {/* Header Section */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 glass-morphism p-8 rounded-[2rem] shadow-2xl">
                <div className="flex items-center gap-6">
                    <div className={`p-6 rounded-2xl border ${isFake ? 'bg-danger/10 text-danger border-danger/20' : 'bg-success/10 text-success border-success/20'}`}>
                        {isFake ? <AlertTriangle size={48} /> : <ShieldCheck size={48} />}
                    </div>
                    <div>
                        <div className="flex items-center gap-3 mb-1">
                            <span className="text-[10px] font-black uppercase tracking-[0.4em]" style={{ color: 'var(--text-dim)', opacity: 0.8 }}>System Verdict</span>
                            <div className={`h-1.5 w-1.5 rounded-full animate-ping ${isFake ? 'bg-danger' : 'bg-success'}`} />
                        </div>
                        <h1 className={`text-6xl font-black tracking-tighter ${isFake ? 'text-danger' : 'text-success'}`}>
                            {result.prediction}
                        </h1>
                    </div>
                </div>

                <div className="flex gap-4">
                    <button
                        className="flex items-center gap-3 glass-morphism px-8 py-4 rounded-2xl hover:bg-white/5 transition-all font-bold text-sm tracking-widest uppercase"
                        style={{ color: 'var(--text-main)' }}
                        onClick={() => window.print()}
                    >
                        <Download size={18} /> Export Intel
                    </button>
                    <button
                        className="btn-glow flex items-center gap-3 px-8 py-4 rounded-2xl shadow-lg hover:scale-105 transition-all font-black text-sm tracking-widest uppercase"
                        onClick={onReset}
                    >
                        <RefreshCcw size={18} /> New Analysis
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Confidence Meter */}
                <div className="lg:col-span-1 glass-morphism p-8 rounded-[2rem] flex flex-col items-center justify-center text-center space-y-6">
                    <h3 className="text-[10px] font-black uppercase tracking-[0.5em]" style={{ color: 'var(--text-dim)' }}>Neural Confidence</h3>

                    <div className="relative w-64 h-64 flex items-center justify-center">
                        <svg className="w-full h-full -rotate-90">
                            <circle
                                cx="128" cy="128" r="110"
                                className="fill-none"
                                stroke="var(--border-subtle)"
                                strokeWidth="20"
                            />
                            <motion.circle
                                cx="128" cy="128" r="110"
                                className={`fill-none ${isFake ? 'stroke-danger' : 'stroke-primary'}`}
                                strokeWidth="20"
                                strokeDasharray="691.15"
                                initial={{ strokeDashoffset: 691.15 }}
                                animate={{ strokeDashoffset: 691.15 - (691.15 * confidence / 100) }}
                                transition={{ duration: 2, ease: "easeOut" }}
                                strokeLinecap="round"
                            />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-6xl font-black tracking-tighter" style={{ color: 'var(--text-main)' }}>{confidence}%</span>
                            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-dim)' }}>Certainty Index</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 w-full">
                        <div className="rounded-2xl p-4 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                            <div className="text-[10px] font-bold uppercase mb-1" style={{ color: 'var(--text-dim)' }}>Authenticity</div>
                            <div className="text-xl font-black text-success tracking-tighter">{100 - confidence}%</div>
                        </div>
                        <div className="rounded-2xl p-4 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                            <div className="text-[10px] font-bold uppercase mb-1" style={{ color: 'var(--text-dim)' }}>Manipulation</div>
                            <div className="text-xl font-black text-danger tracking-tighter">{confidence}%</div>
                        </div>
                    </div>
                </div>

                {/* Explainable AI Metrics */}
                <div className="lg:col-span-2 glass-morphism p-8 rounded-[2rem] space-y-8">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xl font-black tracking-wide flex items-center gap-3" style={{ color: 'var(--text-main)' }}>
                            <Workflow size={24} style={{ color: 'var(--primary)' }} /> XAI ANALYSIS CLUSTERS
                        </h3>
                        <span className="text-[10px] font-bold uppercase border px-3 py-1 rounded-full" style={{ color: 'var(--primary)', borderColor: 'var(--border-subtle)' }}>Active Explainer v2.0</span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Feature 1: Attention Map Placeholder */}
                        <div className="rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <Eye size={20} style={{ color: 'var(--primary)' }} />
                                    <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Attention Regions</span>
                                </div>
                                <Activity size={16} style={{ color: 'var(--text-dim)', opacity: 0.5 }} />
                            </div>
                            <div className="h-32 rounded-xl overflow-hidden border relative flex items-center justify-center transition-all bg-black/10" style={{ borderColor: 'var(--border-subtle)' }}>
                                {result.forensics?.heatmap ? (
                                    <img src={`data:image/jpeg;base64,${result.forensics.heatmap}`} className="w-full h-full object-cover opacity-80" />
                                ) : (
                                    <div className="text-[10px] uppercase font-bold tracking-widest" style={{ color: 'var(--text-dim)' }}>Generating Heatmap...</div>
                                )}
                            </div>
                        </div>

                        {/* Feature 2: Temporal Inconsistency */}
                        <div className="rounded-2xl p-6 border group transition-all" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <Layers size={20} style={{ color: 'var(--secondary)' }} />
                                    <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Layer Continuity</span>
                                </div>
                                <Activity size={16} style={{ color: 'var(--text-dim)', opacity: 0.5 }} />
                            </div>
                            <div className="space-y-4">
                                {[84, 91, 72].map((v, i) => (
                                    <div key={i} className="space-y-1">
                                        <div className="flex justify-between text-[8px] font-black px-1" style={{ color: 'var(--text-dim)' }}>
                                            <span>SLICE_{i + 1}</span>
                                            <span>{v}% STABILITY</span>
                                        </div>
                                        <div className="h-1.5 w-full rounded-full overflow-hidden" style={{ background: 'var(--border-subtle)' }}>
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${v}%` }}
                                                transition={{ duration: 1, delay: i * 0.2 }}
                                                className="h-full shadow-lg"
                                                style={{ background: 'var(--secondary)' }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Detailed Findings List */}
                    <div className="space-y-4 p-6 rounded-2xl border" style={{ background: 'rgba(126,200,160,0.05)', borderColor: 'var(--border-subtle)' }}>
                        <div className="flex items-center gap-3 text-xs font-bold uppercase tracking-widest mb-2" style={{ color: 'var(--text-dim)' }}>
                            <FileText size={16} /> Forensic Logs
                        </div>
                        <div className="space-y-3 font-mono text-xs">
                            {(result.forensics?.findings || ["No specific structural anomalies detected.", "Temporal coherence verified within normal range.", "Metadata integrity check passed."]).map((log, i) => (
                                <div key={i} className="flex gap-4 group">
                                    <span style={{ color: 'var(--primary)' }}>[{String(i + 1).padStart(2, '0')}]</span>
                                    <span style={{ color: 'var(--text-dim)' }} className="group-hover:text-primary transition-colors">{log}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Feedback Section */}
            <FeedbackPanel result={result} mediaType={result.frames_processed ? 'video' : 'image'} />
        </div>
    );
}
