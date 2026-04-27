import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, MessageSquare, Send, CheckCircle, X } from 'lucide-react';
import axios from 'axios';

const API_BASE = "https://deepfake-detection-1-l61c.onrender.com";

export default function FeedbackPanel({ result, mediaType = 'image' }) {
    const [vote, setVote] = useState(null); // 'correct' | 'wrong'
    const [actualLabel, setActualLabel] = useState('');
    const [comments, setComments] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [loading, setLoading] = useState(false);
    const [expanded, setExpanded] = useState(false);

    const handleSubmit = async () => {
        if (vote === null) return;
        setLoading(true);
        try {
            const fd = new FormData();
            fd.append('prediction_correct', vote === 'correct' ? 'true' : 'false');
            fd.append('actual_label', actualLabel || (vote === 'correct' ? result?.prediction : (result?.prediction === 'REAL' ? 'FAKE' : 'REAL')));
            fd.append('comments', comments);
            fd.append('media_type', mediaType);
            fd.append('model_prediction', result?.prediction || '');
            await axios.post(`${API_BASE}/submit_feedback`, fd);
            setSubmitted(true);
        } catch (e) {
            console.error('Feedback error:', e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="w-full mt-6"
        >
            <div className="glass-morphism rounded-[2rem] border overflow-hidden" style={{ borderColor: 'var(--border-subtle)' }}>
                {/* Header toggle */}
                <button
                    onClick={() => setExpanded(p => !p)}
                    className="w-full flex items-center justify-between px-8 py-5 transition-all text-left"
                    style={{ background: 'transparent' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(126,200,160,0.05)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                    <div className="flex items-center gap-3">
                        <MessageSquare size={18} style={{ color: 'var(--primary)' }} />
                        <span className="text-xs font-black uppercase tracking-[0.25em]" style={{ color: 'var(--text-main)' }}>
                            Was this prediction accurate?
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        {submitted && (
                            <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-success">
                                <CheckCircle size={14} /> Submitted
                            </span>
                        )}
                        <motion.div
                            animate={{ rotate: expanded ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                            className="opacity-40"
                            style={{ color: 'var(--text-dim)' }}
                        >
                            ▼
                        </motion.div>
                    </div>
                </button>

                <AnimatePresence>
                    {expanded && !submitted && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                        >
                            <div className="px-8 pb-8 space-y-6 border-t pt-6" style={{ borderColor: 'var(--border-subtle)' }}>

                                {/* Verdict */}
                                <div className="space-y-3">
                                    <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-dim)', opacity: 0.8 }}>
                                        The model predicted: <span className={`font-black ${result?.prediction === 'REAL' ? 'text-success' : 'text-danger'}`}>{result?.prediction}</span>
                                    </p>
                                    <p className="text-xs font-bold" style={{ color: 'var(--text-dim)' }}>Was this correct?</p>
                                    <div className="flex gap-4">
                                        <motion.button
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                            onClick={() => setVote('correct')}
                                            className="flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl border font-black text-xs uppercase tracking-widest transition-all"
                                            style={{
                                                background: vote === 'correct' ? 'rgba(52, 211, 153, 0.15)' : 'var(--bg-card)',
                                                borderColor: vote === 'correct' ? 'rgba(52, 211, 153, 0.5)' : 'var(--border-subtle)',
                                                color: vote === 'correct' ? 'var(--success)' : 'var(--text-dim)',
                                                boxShadow: vote === 'correct' ? '0 0 20px rgba(52, 211, 153, 0.15)' : 'none'
                                            }}
                                        >
                                            <ThumbsUp size={16} /> Yes, Correct
                                        </motion.button>
                                        <motion.button
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                            onClick={() => setVote('wrong')}
                                            className="flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl border font-black text-xs uppercase tracking-widest transition-all"
                                            style={{
                                                background: vote === 'wrong' ? 'rgba(251, 113, 133, 0.15)' : 'var(--bg-card)',
                                                borderColor: vote === 'wrong' ? 'rgba(251, 113, 133, 0.5)' : 'var(--border-subtle)',
                                                color: vote === 'wrong' ? 'var(--danger)' : 'var(--text-dim)',
                                                boxShadow: vote === 'wrong' ? '0 0 20px rgba(251, 113, 133, 0.15)' : 'none'
                                            }}
                                        >
                                            <ThumbsDown size={16} /> No, Wrong
                                        </motion.button>
                                    </div>
                                </div>

                                {/* If wrong — what was actual */}
                                <AnimatePresence>
                                    {vote === 'wrong' && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="space-y-3 overflow-hidden"
                                        >
                                            <p className="text-xs font-bold pt-2" style={{ color: 'var(--text-dim)' }}>What was the actual content?</p>
                                            <div className="flex gap-3">
                                                {['REAL', 'FAKE'].map(label => (
                                                    <button
                                                        key={label}
                                                        onClick={() => setActualLabel(label)}
                                                        className="flex-1 py-3 rounded-xl border text-xs font-black uppercase tracking-widest transition-all"
                                                        style={{
                                                            background: actualLabel === label ? 'rgba(126, 200, 160, 0.15)' : 'var(--bg-card)',
                                                            borderColor: actualLabel === label ? 'var(--primary)' : 'var(--border-subtle)',
                                                            color: actualLabel === label ? 'var(--primary)' : 'var(--text-dim)',
                                                        }}
                                                    >
                                                        {label}
                                                    </button>
                                                ))}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>

                                {/* Comments */}
                                <div className="space-y-2">
                                    <p className="text-xs font-bold" style={{ color: 'var(--text-dim)' }}>Additional comments <span className="opacity-40">(optional)</span></p>
                                    <textarea
                                        value={comments}
                                        onChange={e => setComments(e.target.value)}
                                        placeholder="e.g. 'The image was clearly real but model said fake...'"
                                        rows={3}
                                        className="w-full border rounded-2xl px-4 py-3 text-sm font-mono resize-none focus:outline-none transition-colors"
                                        style={{
                                            background: 'var(--bg-card)',
                                            borderColor: 'var(--border-subtle)',
                                            color: 'var(--text-main)',
                                        }}
                                        onFocus={e => e.target.style.borderColor = 'var(--primary)'}
                                        onBlur={e => e.target.style.borderColor = 'var(--border-subtle)'}
                                    />
                                </div>

                                {/* Submit */}
                                <motion.button
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={handleSubmit}
                                    disabled={vote === null || loading}
                                    className="w-full flex items-center justify-center gap-3 py-4 rounded-2xl font-black text-xs uppercase tracking-[0.25em] transition-all"
                                    style={{
                                        background: vote !== null ? 'var(--primary)' : 'var(--bg-card)',
                                        borderColor: vote !== null ? 'var(--primary)' : 'var(--border-subtle)',
                                        color: vote !== null ? '#080d0a' : 'var(--text-dim)',
                                        opacity: vote !== null ? 1 : 0.5,
                                        cursor: vote !== null ? 'pointer' : 'not-allowed',
                                        boxShadow: vote !== null ? '0 0 20px var(--glow-primary)' : 'none'
                                    }}
                                >
                                    {loading ? (
                                        <span className="flex items-center gap-2">
                                            <motion.div
                                                animate={{ rotate: 360 }}
                                                transition={{ repeat: Infinity, duration: 0.8, ease: 'linear' }}
                                                className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
                                            />
                                            Submitting...
                                        </span>
                                    ) : (
                                        <>
                                            <Send size={14} /> Submit Feedback
                                        </>
                                    )}
                                </motion.button>
                            </div>
                        </motion.div>
                    )}

                    {expanded && submitted && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="px-8 pb-8 pt-4 border-t"
                            style={{ borderColor: 'var(--border-subtle)' }}
                        >
                            <div className="flex flex-col items-center gap-3 py-6">
                                <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ type: 'spring', stiffness: 200 }}
                                >
                                    <CheckCircle size={48} className="text-success" />
                                </motion.div>
                                <p className="text-sm font-black uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Thank you!</p>
                                <p className="text-xs text-center" style={{ color: 'var(--text-dim)' }}>
                                    Your feedback helps improve the model's accuracy over time.
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </motion.div>
    );
}
