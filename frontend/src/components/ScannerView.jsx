import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
// import { Canvas } from '@react-three/fiber';
// import { MeshDistortMaterial, Sphere, Float } from '@react-three/drei';

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
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-dark/90 backdrop-blur-3xl overflow-hidden">
            {/* Background Grid */}
            <div className="absolute inset-0 opacity-10"
                style={{ backgroundImage: 'linear-gradient(#22d3ee 1px, transparent 1px), linear-gradient(90deg, #22d3ee 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

            {/* Scanning Line */}
            <motion.div
                initial={{ top: '-10%' }}
                animate={{ top: '110%' }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 right-0 h-[2px] bg-primary/50 shadow-[0_0_20px_#22d3ee] z-10"
            />

            {/* 3D Neural Blob */}
            <div className="relative w-full h-80 mb-12">
                {/* <Canvas>
                    <ambientLight intensity={0.5} />
                    <pointLight position={[10, 10, 10]} />
                    <Float speed={5} rotationIntensity={2} floatIntensity={2}>
                        <Sphere args={[1, 100, 100]} scale={2.5}>
                            <MeshDistortMaterial
                                color="#22d3ee"
                                speed={4}
                                distort={0.4}
                                radius={1}
                            />
                        </Sphere>
                    </Float>
                </Canvas> */}

                {/* Progress Circle Overlay */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="text-8xl font-black text-white/10 select-none">
                        {progress}%
                    </div>
                </div>
            </div>

            <div className="relative z-20 text-center space-y-8">
                <h2 className="text-4xl font-black tracking-tighter text-white">
                    SCANNING <span className="text-primary italic uppercase">{modality}</span> SOURCE
                </h2>

                <div className="h-8">
                    <AnimatePresence mode="wait">
                        <motion.p
                            key={messageIndex}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="text-primary font-mono text-lg tracking-widest uppercase"
                        >
                            {"> "} {messages[messageIndex]}
                        </motion.p>
                    </AnimatePresence>
                </div>

                {/* Custom Progress Bar */}
                <div className="w-96 h-1 bg-white/5 rounded-full overflow-hidden border border-white/10">
                    <motion.div
                        className="h-full bg-gradient-to-r from-primary to-secondary shadow-[0_0_15px_#22d3ee]"
                        animate={{ width: `${progress}%` }}
                    />
                </div>

                <div className="flex justify-between items-center text-[10px] font-bold text-dim tracking-[0.3em] uppercase opacity-40">
                    <span>Neural Engine v4.0</span>
                    <span>Security Protocol Enabled</span>
                </div>
            </div>
        </div>
    );
}
