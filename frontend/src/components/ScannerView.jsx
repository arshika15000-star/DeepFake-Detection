import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const messages = [
    "Initializing Neural Link...",
    "Extracting Forensic Metadata...",
    "Analyzing Facial Inconsistencies...",
    "Detecting Synthetic Voice Patterns...",
    "Evaluating Semantic Authenticity...",
    "Aggregating Multi-modal Signatures...",
    "Generating Final Verdict..."
];

export default function ScannerView({ modality, onComplete }) {
    const [progress, setProgress] = useState(0);
    const [messageIndex, setMessageIndex] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    clearInterval(timer);
                    setTimeout(onComplete, 500);
                    return 100;
                }
                return prev + 1;
            });
        }, 50);

        const msgTimer = setInterval(() => {
            setMessageIndex(prev => (prev + 1) % messages.length);
        }, 1500);

        return () => {
            clearInterval(timer);
            clearInterval(msgTimer);
        };
    }, [onComplete]);

    return (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-dark/90 backdrop-blur-3xl overflow-hidden" style={{ background: 'var(--bg-body)' }}>
            {/* Background Grid */}
            <div className="absolute inset-0 opacity-10"
                style={{ backgroundImage: 'linear-gradient(var(--primary) 1px, transparent 1px), linear-gradient(90deg, var(--primary) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

            {/* Scanning Line */}
            <motion.div
                initial={{ top: '-10%' }}
                animate={{ top: '110%' }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 right-0 h-[2px] z-10"
                style={{ background: 'var(--primary)', opacity: 0.5, boxShadow: '0 0 20px var(--glow-primary)' }}
            />

            {/* Progress Circle Overlay */}
            <div className="relative w-full h-80 mb-12">
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="text-8xl font-black select-none" style={{ color: 'var(--primary)', opacity: 0.1 }}>
                        {progress}%
                    </div>
                </div>
            </div>

            <div className="relative z-20 text-center space-y-8">
                <h2 className="text-4xl font-black tracking-tighter" style={{ color: 'var(--text-main)' }}>
                    SCANNING <span className="italic uppercase" style={{ color: 'var(--primary)' }}>{modality}</span> SOURCE
                </h2>

                <div className="h-8">
                    <AnimatePresence mode="wait">
                        <motion.p
                            key={messageIndex}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="font-mono text-lg tracking-widest uppercase"
                            style={{ color: 'var(--primary)' }}
                        >
                            {"> "} {messages[messageIndex]}
                        </motion.p>
                    </AnimatePresence>
                </div>

                {/* Custom Progress Bar */}
                <div className="w-96 h-1 rounded-full overflow-hidden border" style={{ background: 'rgba(126,200,160,0.1)', borderColor: 'var(--border-subtle)' }}>
                    <motion.div
                        className="h-full"
                        style={{ background: 'linear-gradient(90deg, var(--primary), var(--secondary))', boxShadow: '0 0 15px var(--glow-primary)' }}
                        animate={{ width: `${progress}%` }}
                    />
                </div>

                <div className="flex justify-between items-center text-[10px] font-bold tracking-[0.3em] uppercase opacity-40" style={{ color: 'var(--text-dim)' }}>
                    <span>Neural Engine v4.0</span>
                    <span>Security Protocol Enabled</span>
                </div>
            </div>
        </div>
    );
}
