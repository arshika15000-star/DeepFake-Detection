import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
    ShieldCheck, AlertTriangle, Activity, Download, RefreshCcw,
    Layers, Eye, FileText, Workflow, Fingerprint, Mic, Type,
    PlaySquare, Radio, Cpu, ChevronDown, ChevronUp, Table2
} from 'lucide-react';
import FeedbackPanel from './FeedbackPanel';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// ─── Reusable forensic image card ────────────────────────────────────────────
function ForensicImage({ b64, title, subtitle, icon: Icon, accentColor }) {
    if (!b64) return null;
    const color = accentColor || 'var(--primary)';
    return (
        <div className="rounded-2xl p-5 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Icon size={16} style={{ color }} />
                    <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>{title}</span>
                </div>
                {subtitle && (
                    <span className="text-[9px] uppercase font-black px-2 py-0.5 rounded-full border"
                        style={{ color, borderColor: color + '55', background: color + '18' }}>{subtitle}</span>
                )}
            </div>
            <div className="h-44 rounded-xl overflow-hidden border relative bg-black/30" style={{ borderColor: 'var(--border-subtle)' }}>
                <img
                    src={`data:image/jpeg;base64,${b64}`}
                    className="w-full h-full object-cover hover:scale-105 transition-transform duration-500"
                    alt={title}
                />
            </div>
        </div>
    );
}

// ─── Probability bar row ──────────────────────────────────────────────────────
function ProbBar({ label, value, color, delay = 0 }) {
    const pct = Math.round((value || 0) * 100);
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-[9px] font-black uppercase px-0.5" style={{ color: 'var(--text-dim)' }}>
                <span>{label}</span><span>{pct}%</span>
            </div>
            <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid var(--border-subtle)' }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.9, delay, ease: 'easeOut' }}
                    className="h-full rounded-full"
                    style={{ background: color }}
                />
            </div>
        </div>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function ResultDashboard({ result, onReset }) {
    const [isExporting, setIsExporting] = useState(false);
    const [showAll, setShowAll] = useState(false);
    const [showMeta, setShowMeta] = useState(false);

    const isFake = result.prediction === 'FAKE';
    const confidence = Math.round(result.confidence * 100);
    const finalPrediction = result.prediction;

    // Infer media type
    const isVideo = !!result.frames_processed;
    const isAudio = result.forensics?.vocal_jitter !== undefined;
    const isText  = result.forensics?.complexity_index !== undefined;
    const isImage = !isVideo && !isAudio && !isText;
    const mediaType = isVideo ? 'video' : isAudio ? 'audio' : isText ? 'text' : 'image';

    const forensics = result.forensics || {};
    const findings  = forensics.findings || result.metadata?.findings || forensics.metadata?.findings || [];
    const metadata  = forensics.metadata || {};
    const metaFindings = metadata.findings || [];
    const sourceAttrib = forensics.source_attribution;

    const probabilities = result.probabilities || {
        fake: isFake ? result.confidence : 1 - result.confidence,
        real: isFake ? 1 - result.confidence : result.confidence,
    };

    const pct = (val) => Math.round((val || 0) * 100);

    // ─── PDF Export ────────────────────────────────────────────────────────────
    const downloadReport = async () => {
        setIsExporting(true);
        try {
            const el = document.getElementById('report-content');
            await new Promise(r => setTimeout(r, 50));
            const canvas = await html2canvas(el, { scale: 2, useCORS: true, backgroundColor: '#0a0f1a' });
            const imgData = canvas.toDataURL('image/jpeg', 0.92);
            const pdf = new jsPDF('p', 'mm', 'a4');
            const pw = pdf.internal.pageSize.getWidth();
            const ph = (canvas.height * pw) / canvas.width;
            pdf.addImage(imgData, 'JPEG', 0, 0, pw, Math.min(ph, 297));

            // Page 2: text summary
            pdf.addPage();
            pdf.setFontSize(18); pdf.setTextColor(126, 200, 160);
            pdf.text('DeepTruth AI — Forensic Analysis Report', 14, 20);
            pdf.setFontSize(10); pdf.setTextColor(200, 200, 200);
            pdf.text(`Generated: ${new Date().toLocaleString()}`, 14, 30);
            pdf.text(`Modality: ${mediaType.toUpperCase()}   |   Verdict: ${finalPrediction}   |   Confidence: ${confidence}%`, 14, 38);
            pdf.setDrawColor(126, 200, 160); pdf.line(14, 42, 196, 42);

            pdf.setFontSize(12); pdf.setTextColor(255, 255, 255);
            pdf.text('Forensic Execution Logs:', 14, 52);
            pdf.setFontSize(9); pdf.setTextColor(180, 210, 195);
            let y = 60;
            findings.slice(0, 40).forEach((f, i) => {
                const lines = pdf.splitTextToSize(`[${String(i + 1).padStart(2, '0')}] ${f}`, 182);
                if (y + lines.length * 5 > 285) { pdf.addPage(); y = 20; }
                pdf.text(lines, 14, y);
                y += lines.length * 5 + 1;
            });

            if (metadata.software && metadata.software !== 'Unknown') {
                if (y + 20 > 285) { pdf.addPage(); y = 20; }
                pdf.setFontSize(12); pdf.setTextColor(255, 255, 255);
                pdf.text('Metadata:', 14, y + 8); y += 16;
                pdf.setFontSize(9); pdf.setTextColor(180, 210, 195);
                pdf.text(`Software: ${metadata.software}`, 14, y); y += 6;
                if (metadata.creation_date && metadata.creation_date !== 'Unknown') {
                    pdf.text(`Created: ${metadata.creation_date}`, 14, y);
                }
            }

            pdf.save(`DeepTruth_Report_${Date.now()}.pdf`);
        } catch (err) {
            console.error(err);
            alert('PDF export failed: ' + err.message);
        } finally {
            setIsExporting(false);
        }
    };

    // ─── Render ───────────────────────────────────────────────────────────────
    return (
        <div id="report-content" className="min-h-screen pt-20 pb-12 px-6 lg:px-12 max-w-7xl mx-auto space-y-8 fade-up" style={{ color: 'var(--text-main)' }}>

            {/* ── HEADER VERDICT CARD ── */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 glass-morphism p-8 rounded-[2rem] shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 rounded-full blur-3xl -z-10 translate-x-1/2 -translate-y-1/2"
                    style={{ background: isFake ? 'rgba(239,68,68,0.08)' : 'rgba(126,200,160,0.08)' }} />
                <div className="flex items-center gap-6 z-10">
                    <div className={`p-5 rounded-3xl border shadow-xl ${isFake ? 'bg-danger/10 text-danger border-danger/20' : 'bg-success/10 text-success border-success/20'}`}>
                        {isFake ? <AlertTriangle size={52} /> : <ShieldCheck size={52} />}
                    </div>
                    <div>
                        <div className="flex items-center gap-3 mb-1">
                            <span className="text-[10px] font-black uppercase tracking-[0.4em]" style={{ color: 'var(--text-dim)' }}>
                                {mediaType} Verdict
                            </span>
                            <div className={`h-2 w-2 rounded-full animate-ping ${isFake ? 'bg-danger' : 'bg-success'}`} />
                        </div>
                        <h1 className={`text-6xl max-md:text-4xl font-black tracking-tighter ${isFake ? 'text-danger' : 'text-success'}`}>
                            {finalPrediction}
                        </h1>
                        {isFake && sourceAttrib && (
                            <div className="mt-2 text-xs font-bold px-3 py-1 rounded-full bg-danger/10 text-danger border border-danger/20 inline-flex items-center gap-2">
                                <Fingerprint size={12} />
                                Predicted: {sourceAttrib.most_likely} ({pct(sourceAttrib.confidence)}%)
                            </div>
                        )}
                    </div>
                </div>

                <div className="flex gap-3 z-10 w-full md:w-auto" data-html2canvas-ignore="true">
                    <button
                        onClick={downloadReport}
                        disabled={isExporting}
                        className="flex-1 md:flex-none flex items-center justify-center gap-2 glass-morphism px-6 py-3 rounded-2xl hover:bg-white/5 transition-all font-bold text-sm tracking-widest uppercase border"
                        style={{ color: 'var(--text-main)', borderColor: 'var(--border-subtle)' }}>
                        <Download size={16} /> {isExporting ? 'Exporting…' : 'Export PDF'}
                    </button>
                    <button
                        onClick={onReset}
                        className="flex-1 md:flex-none btn-glow flex items-center justify-center gap-2 px-6 py-3 rounded-2xl shadow-lg hover:scale-105 transition-all font-black text-sm tracking-widest uppercase">
                        <RefreshCcw size={16} /> New Analysis
                    </button>
                </div>
            </div>

            {/* ── MAIN CONTENT GRID ── */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* ── LEFT: Confidence + Probability Breakdown ── */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Circular confidence meter */}
                    <div className="glass-morphism p-8 rounded-[2rem] flex flex-col items-center text-center space-y-6 relative overflow-hidden">
                        <h3 className="text-[10px] font-black uppercase tracking-[0.5em]" style={{ color: 'var(--text-dim)' }}>Neural Confidence</h3>
                        <div className="relative w-52 h-52 flex items-center justify-center">
                            <svg className="w-full h-full -rotate-90">
                                <circle cx="104" cy="104" r="90" className="fill-none" stroke="var(--border-subtle)" strokeWidth="14" />
                                <motion.circle
                                    cx="104" cy="104" r="90"
                                    fill="none"
                                    className={isFake ? 'stroke-danger' : 'stroke-primary'}
                                    strokeWidth="14"
                                    strokeDasharray="565.49"
                                    initial={{ strokeDashoffset: 565.49 }}
                                    animate={{ strokeDashoffset: 565.49 - (565.49 * confidence / 100) }}
                                    transition={{ duration: 1.5, ease: 'easeOut' }}
                                    strokeLinecap="round"
                                />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-5xl font-black tracking-tighter" style={{ color: 'var(--text-main)' }}>{confidence}%</span>
                                <span className="text-[9px] font-bold uppercase tracking-widest mt-1" style={{ color: 'var(--text-dim)' }}>Certainty Index</span>
                            </div>
                        </div>

                        {/* Probability bars */}
                        <div className="w-full space-y-3">
                            <ProbBar label="Fake Probability" value={probabilities.fake} color="var(--danger)" delay={0.1} />
                            <ProbBar label="Real Probability" value={probabilities.real} color="var(--success)" delay={0.3} />
                        </div>
                    </div>

                    {/* Source Attribution */}
                    {sourceAttrib?.all_probs && (
                        <div className="glass-morphism p-6 rounded-[2rem] space-y-4">
                            <div className="flex items-center gap-2">
                                <Fingerprint size={16} style={{ color: 'var(--danger)' }} />
                                <span className="text-xs font-black uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>AI Generator Fingerprint</span>
                            </div>
                            <div className="space-y-3">
                                {Object.entries(sourceAttrib.all_probs)
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([name, prob], i) => (
                                        <ProbBar key={name} label={name} value={prob}
                                            color={i === 0 ? 'var(--danger)' : 'var(--primary)'}
                                            delay={i * 0.08} />
                                    ))
                                }
                            </div>
                        </div>
                    )}
                </div>

                {/* ── RIGHT: XAI Artifacts + Findings ── */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="glass-morphism p-8 rounded-[2rem] space-y-6">
                        {/* Section title */}
                        <div className="flex justify-between items-center border-b pb-4" style={{ borderColor: 'var(--border-subtle)' }}>
                            <h3 className="text-xl font-black tracking-wide flex items-center gap-3" style={{ color: 'var(--text-main)' }}>
                                <Workflow size={22} style={{ color: 'var(--primary)' }} /> XAI ANALYSIS CLUSTERS
                            </h3>
                            <span className="text-[10px] font-bold uppercase border px-3 py-1 rounded-full bg-primary/10"
                                style={{ color: 'var(--primary)', borderColor: 'var(--primary)' }}>
                                {mediaType.toUpperCase()} EXPLAINER
                            </span>
                        </div>

                        {/* ── IMAGE artifacts: Grad-CAM, ELA, FFT, Noise ── */}
                        {(isImage || (isVideo && forensics.heatmap)) && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <ForensicImage b64={forensics.heatmap} title="Attention Map (Grad-CAM)" subtitle="XAI" icon={Eye} />
                                {isImage && <ForensicImage b64={forensics.ela} title="Error Level Analysis" subtitle="ELA" icon={Layers} accentColor="var(--secondary)" />}
                                {isImage && <ForensicImage b64={forensics.fft} title="FFT Frequency Map" subtitle="Spectral" icon={Radio} accentColor="#ff9063" />}
                                {isImage && <ForensicImage b64={forensics.noise} title="Noise Print" subtitle="High-Freq" icon={Cpu} accentColor="#c87eff" />}
                            </div>
                        )}

                        {/* ── VIDEO: Neural metrics + rPPG Pulse ── */}
                        {isVideo && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                {forensics.neural_metrics && (
                                    <div className="rounded-2xl p-5 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                        <div className="flex items-center gap-2 mb-4">
                                            <PlaySquare size={16} style={{ color: 'var(--secondary)' }} />
                                            <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>Neural Metrics</span>
                                        </div>
                                        <div className="space-y-3">
                                            {Object.entries(forensics.neural_metrics).map(([key, val], i) => (
                                                <div key={key} className="space-y-1">
                                                    <div className="flex justify-between text-[9px] font-black" style={{ color: 'var(--text-dim)' }}>
                                                        <span>{key.replace(/_/g, ' ').toUpperCase()}</span><span>{pct(val)}%</span>
                                                    </div>
                                                    <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                                                        <motion.div initial={{ width: 0 }} animate={{ width: `${pct(val)}%` }}
                                                            transition={{ duration: 1, delay: i * 0.15 }}
                                                            className="h-full"
                                                            style={{ background: 'linear-gradient(90deg,var(--primary),var(--secondary))' }} />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {result.pulse_data && (
                                    <div className="rounded-2xl p-5 border flex flex-col" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                        <div className="flex items-center gap-2 mb-4">
                                            <Activity size={16} style={{ color: 'var(--danger)' }} />
                                            <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>rPPG Neural Pulse</span>
                                        </div>
                                        <div className="flex-1 flex items-end justify-between gap-0.5 h-28">
                                            {result.pulse_data.slice(0, 30).map((p, i) => (
                                                <motion.div key={i} className="w-full rounded-t"
                                                    style={{ background: 'var(--primary)', opacity: 0.8 }}
                                                    initial={{ height: 0 }}
                                                    animate={{ height: `${(p / 100) * 100}%` }}
                                                    transition={{ duration: 0.5, delay: i * 0.02 }} />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* ── AUDIO: Jitter + Spectral floor ── */}
                        {isAudio && (
                            <div className="rounded-2xl p-5 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center gap-2 mb-4">
                                    <Mic size={16} style={{ color: 'var(--primary)' }} />
                                    <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>Acoustic Fingerprint Analysis</span>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-black/20 p-4 rounded-xl space-y-2 border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase" style={{ color: 'var(--text-dim)' }}>Vocal Jitter Factor</div>
                                        <div className="text-3xl font-black" style={{ color: 'var(--primary)' }}>{pct(forensics.vocal_jitter || 0)}%</div>
                                        <div className="text-[10px]" style={{ color: 'var(--text-dim)' }}>Micro-fluctuations in pitch</div>
                                        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                                            <motion.div initial={{ width: 0 }} animate={{ width: `${pct(forensics.vocal_jitter || 0)}%` }}
                                                transition={{ duration: 1 }} className="h-full" style={{ background: 'var(--primary)' }} />
                                        </div>
                                    </div>
                                    <div className="bg-black/20 p-4 rounded-xl space-y-2 border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase" style={{ color: 'var(--text-dim)' }}>Spectral Floor</div>
                                        <div className="text-sm font-bold uppercase mt-2" style={{ color: 'var(--secondary)' }}>{forensics.spectral_floor || 'Unknown'}</div>
                                        <div className="text-[10px]" style={{ color: 'var(--text-dim)' }}>Background noise profile</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ── TEXT: LLM analysis ── */}
                        {isText && (
                            <div className="rounded-2xl p-5 border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}>
                                <div className="flex items-center gap-2 mb-4">
                                    <Type size={16} style={{ color: 'var(--primary)' }} />
                                    <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-main)' }}>LLM Linguistic Structure Analysis</span>
                                </div>
                                {forensics.llm_summary && (
                                    <div className="mb-4 bg-primary/10 border p-4 rounded-xl flex items-start gap-3" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <Workflow size={16} style={{ color: 'var(--primary)', marginTop: 2, flexShrink: 0 }} />
                                        <p className="text-sm font-semibold leading-relaxed" style={{ color: 'var(--text-main)' }}>{forensics.llm_summary}</p>
                                    </div>
                                )}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-black/20 p-4 rounded-xl border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase mb-1" style={{ color: 'var(--text-dim)' }}>Complexity Index</div>
                                        <div className="text-3xl font-black" style={{ color: 'var(--primary)' }}>{(forensics.complexity_index || 0).toFixed(2)}</div>
                                        <div className="text-[10px] mt-1" style={{ color: 'var(--text-dim)' }}>Vocabulary richness & sentence depth</div>
                                    </div>
                                    <div className="bg-black/20 p-4 rounded-xl border" style={{ borderColor: 'var(--border-subtle)' }}>
                                        <div className="text-[10px] font-bold uppercase mb-1" style={{ color: 'var(--text-dim)' }}>Structural Rigidity</div>
                                        <div className={`text-sm font-bold mt-2 uppercase ${forensics.is_structured ? 'text-danger' : 'text-success'}`}>
                                            {forensics.is_structured ? 'High (Machine-like)' : 'Normal (Human-like)'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ── FORENSIC METADATA TABLE ── */}
                        {(isImage || isVideo) && (metadata.software || metadata.creation_date || metaFindings.length > 0) && (
                            <div className="rounded-2xl border overflow-hidden" style={{ borderColor: 'var(--border-subtle)' }}>
                                <button
                                    onClick={() => setShowMeta(v => !v)}
                                    className="w-full flex items-center justify-between px-5 py-3 text-xs font-black uppercase tracking-widest hover:bg-white/5 transition-colors"
                                    style={{ background: 'var(--bg-card)', color: metadata.suspicious ? 'var(--danger)' : 'var(--primary)' }}>
                                    <div className="flex items-center gap-2">
                                        <Table2 size={14} />
                                        Forensic Metadata
                                        {metadata.suspicious && (
                                            <span className="ml-2 px-2 py-0.5 rounded-full text-[9px] bg-danger/20 text-danger border border-danger/30">⚠ AI Signature</span>
                                        )}
                                    </div>
                                    {showMeta ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                </button>
                                {showMeta && (
                                    <div className="px-5 py-4 space-y-2 font-mono text-xs" style={{ background: 'rgba(0,0,0,0.3)' }}>
                                        {metadata.software && metadata.software !== 'Unknown' && (
                                            <div className="flex gap-3">
                                                <span style={{ color: 'var(--text-dim)', minWidth: 110 }} className="font-bold">Software:</span>
                                                <span style={{ color: 'var(--text-main)' }}>{metadata.software}</span>
                                            </div>
                                        )}
                                        {metadata.creation_date && metadata.creation_date !== 'Unknown' && (
                                            <div className="flex gap-3">
                                                <span style={{ color: 'var(--text-dim)', minWidth: 110 }} className="font-bold">Created:</span>
                                                <span style={{ color: 'var(--text-main)' }}>{metadata.creation_date}</span>
                                            </div>
                                        )}
                                        {metaFindings.length > 0 && (
                                            <div className="pt-2 space-y-1">
                                                <div className="font-bold mb-1" style={{ color: 'var(--text-dim)' }}>Metadata Findings:</div>
                                                {metaFindings.map((f, i) => (
                                                    <div key={i} className="flex gap-2">
                                                        <span style={{ color: metadata.suspicious ? 'var(--danger)' : 'var(--primary)' }} className="flex-shrink-0">
                                                            [{String(i + 1).padStart(2, '0')}]
                                                        </span>
                                                        <span style={{ color: 'var(--text-dim)' }}>{f}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* ── FORENSIC EXECUTION LOGS ── */}
                        <div className="rounded-2xl border p-5 space-y-3" style={{ background: 'rgba(126,200,160,0.04)', borderColor: 'var(--border-subtle)' }}>
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest" style={{ color: 'var(--text-dim)' }}>
                                    <FileText size={15} style={{ color: 'var(--primary)' }} />
                                    Forensic Execution Logs
                                    <span className="px-2 py-0.5 rounded-full text-[9px] font-black"
                                        style={{ background: 'var(--primary)', color: '#000' }}>{findings.length}</span>
                                </div>
                                {findings.length > 5 && (
                                    <button
                                        onClick={() => setShowAll(v => !v)}
                                        className="text-[10px] font-black uppercase tracking-widest flex items-center gap-1 hover:opacity-70 transition-opacity"
                                        style={{ color: 'var(--primary)' }}>
                                        {showAll ? <><ChevronUp size={12} /> Show Less</> : <><ChevronDown size={12} /> Show All ({findings.length})</>}
                                    </button>
                                )}
                            </div>
                            <div
                                className="space-y-2 font-mono text-xs overflow-y-auto pr-1 custom-scrollbar"
                                style={{ maxHeight: showAll ? '320px' : '110px', transition: 'max-height 0.35s ease' }}>
                                {findings.length > 0 ? findings.map((log, i) => (
                                    <div key={i} className="flex gap-3 items-start group">
                                        <span style={{ color: 'var(--primary)' }} className="flex-shrink-0">[{String(i + 1).padStart(2, '0')}]</span>
                                        <span style={{ color: 'var(--text-dim)' }} className="group-hover:text-primary transition-colors leading-relaxed">{log}</span>
                                    </div>
                                )) : (
                                    <div className="flex gap-3 items-start">
                                        <span style={{ color: 'var(--primary)' }}>[01]</span>
                                        <span style={{ color: 'var(--text-dim)' }}>No forensic anomalies registered for this sequence.</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── FEEDBACK ── */}
            <div data-html2canvas-ignore="true">
                <FeedbackPanel result={result} mediaType={mediaType} />
            </div>
        </div>
    );
}
