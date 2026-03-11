import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, MessageSquare, Send, CheckCircle, X } from 'lucide-react';
import axios from 'axios';

const API_BASE = "http://127.0.0.1:8000";

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
            <div className="glass-morphism rounded-[2rem] border border-white/5 overflow-hidden">
                {/* Header toggle */}
                <button
                    onClick={() => setExpanded(p => !p)}
                    className="w-full flex items-center justify-between px-8 py-5 hover:bg-white/5 transition-all"
                >
                    <div className="flex items-center gap-3">
                        <MessageSquare size={18} className="text-primary" />
                        <span className="text-xs font-black uppercase tracking-[0.25em] text-white">
                            Was this prediction accurate?
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        {submitted && (
                            <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-green-400">
                                <CheckCircle size={14} /> Submitted
                            </span>
                        )}
                        <motion.div
                            animate={{ rotate: expanded ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                            className="text-dim opacity-40"
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
                            <div className="px-8 pb-8 space-y-6 border-t border-white/5 pt-6">

                                {/* Verdict */}
                                <div className="space-y-3">
                                    <p className="text-[10px] font-bold uppercase tracking-widest text-dim opacity-60">
                                        The model predicted: <span className={`font-black ${result?.prediction === 'REAL' ? 'text-green-400' : 'text-red-400'}`}>{result?.prediction}</span>
                                    </p>
                                    <p className="text-xs font-bold text-white/60">Was this correct?</p>
                                    <div className="flex gap-4">
                                        <motion.button
                                            whileHover={{ scale: 1.05 }}
                                            whileTap={{ scale: 0.95 }}
                                            onClick={() => setVote('correct')}
                                            className={`flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl border font-black text-xs uppercase tracking-widest transition-all ${
                                                vote === 'correct'
                                                    ? 'bg-green-500/20 border-green-500/60 text-green-400 shadow-[0_0_20px_rgba(74,222,128,0.2)]'
                                                    : 'bg-white/5 border-white/10 text-dim hover:border-green-500/30'
                                            }`}
                                        >
                                            <ThumbsUp size={16} /> Yes, Correct
                                        </motion.button>
                                        <motion.button
                                            whileHover={{ scale: 1.05 }}
                                            whileTap={{ scale: 0.95 }}
                                            onClick={() => setVote('wrong')}
                                            className={`flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl border font-black text-xs uppercase tracking-widest transition-all ${
                                                vote === 'wrong'
                                                    ? 'bg-red-500/20 border-red-500/60 text-red-400 shadow-[0_0_20px_rgba(248,113,113,0.2)]'
                                                    : 'bg-white/5 border-white/10 text-dim hover:border-red-500/30'
                                            }`}
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
                                            className="space-y-3"
                                        >
                                            <p className="text-xs font-bold text-white/60">What was the actual content?</p>
                                            <div className="flex gap-3">
                                                {['REAL', 'FAKE'].map(label => (
                                                    <button
                                                        key={label}
                                                        onClick={() => setActualLabel(label)}
                                                        className={`flex-1 py-3 rounded-xl border text-xs font-black uppercase tracking-widest transition-all ${
                                                            actualLabel === label
                                                                ? 'bg-primary/20 border-primary/60 text-primary'
                                                                : 'bg-white/5 border-white/10 text-dim hover:border-primary/30'
                                                        }`}
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
                                    <p className="text-xs font-bold text-white/60">Additional comments <span className="opacity-40">(optional)</span></p>
                                    <textarea
                                        value={comments}
                                        onChange={e => setComments(e.target.value)}
                                        placeholder="e.g. 'The image was clearly real but model said fake...'"
                                        rows={3}
                                        className="w-full bg-white/5 border border-white/10 rounded-2xl px-4 py-3 text-sm text-white placeholder:text-white/20 font-mono resize-none focus:outline-none focus:border-primary/40 transition-colors"
                                    />
                                </div>

                                {/* Submit */}
                                <motion.button
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={handleSubmit}
                                    disabled={vote === null || loading}
                                    className={`w-full flex items-center justify-center gap-3 py-4 rounded-2xl font-black text-xs uppercase tracking-[0.25em] transition-all ${
                                        vote !== null
                                            ? 'bg-primary/20 border border-primary/40 text-primary hover:bg-primary hover:text-dark shadow-[0_0_30px_rgba(34,211,238,0.15)]'
                                            : 'bg-white/5 border border-white/10 text-white/20 cursor-not-allowed'
                                    }`}
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
                            className="px-8 pb-8 pt-4 border-t border-white/5"
                        >
                            <div className="flex flex-col items-center gap-3 py-6">
                                <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ type: 'spring', stiffness: 200 }}
                                >
                                    <CheckCircle size={48} className="text-green-400" />
                                </motion.div>
                                <p className="text-sm font-black uppercase tracking-widest text-white">Thank you!</p>
                                <p className="text-xs text-dim opacity-50 text-center">
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
