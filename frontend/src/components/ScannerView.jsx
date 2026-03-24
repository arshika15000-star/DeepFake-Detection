import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ScannerView({ modality, progress }) {
    const percent = progress?.percent || 0;
    const status = progress?.status || "initializing";

    // Format the backend status into a readable message
    const formattedStatus = status.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') + "...";

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
                        {percent}%
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
                            key={formattedStatus}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="font-mono text-lg tracking-widest uppercase"
                            style={{ color: 'var(--primary)' }}
                        >
                            {"> "} {formattedStatus}
                        </motion.p>
                    </AnimatePresence>
                </div>

                {/* Custom Progress Bar */}
                <div className="w-96 h-1 rounded-full overflow-hidden border" style={{ background: 'rgba(126,200,160,0.1)', borderColor: 'var(--border-subtle)' }}>
                    <motion.div
                        className="h-full shadow-lg"
                        style={{ background: 'linear-gradient(90deg, var(--primary), var(--secondary))', boxShadow: '0 0 15px var(--glow-primary)' }}
                        animate={{ width: `${percent}%` }}
                    />
                </div>

                <div className="flex justify-between items-center text-[10px] font-bold tracking-[0.3em] uppercase opacity-40" style={{ color: 'var(--text-dim)' }}>
                    <span>Neural Engine v4.0</span>
                    <span>Live Telemetry Streams</span>
                </div>
            </div>
        </div>
    );
}

