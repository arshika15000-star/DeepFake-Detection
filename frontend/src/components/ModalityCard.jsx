import React from 'react';
import { motion } from 'framer-motion';

export default function ModalityCard({ title, icon, description, onClick, color = "primary", isDark }) {
    // Dynamic styles based on theme rather than hardcoding Tailwind colors everywhere
    const borderColor = color === 'primary' ? 'var(--primary)' : 'var(--secondary)';
    const glowColor = color === 'primary' ? 'var(--glow-primary)' : 'var(--glow-secondary)';

    return (
        <motion.div
            whileHover={{ scale: 1.05, rotateY: 5, rotateX: -5, y: -5 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            onClick={onClick}
            className="group relative glass-morphism p-8 rounded-3xl cursor-pointer transition-all duration-500 overflow-hidden"
            style={{
                borderColor: 'var(--border-subtle)'
            }}
            onMouseEnter={e => {
                e.currentTarget.style.borderColor = borderColor;
                e.currentTarget.style.boxShadow = `0 10px 30px ${glowColor}`;
                e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.95)';
            }}
            onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'var(--border-subtle)';
                e.currentTarget.style.boxShadow = 'none';
                e.currentTarget.style.background = 'var(--bg-card)';
            }}
        >
            <div className="relative z-10 flex flex-col items-center text-center">
                <div 
                    className="text-5xl mb-5 transition-transform group-hover:scale-110 group-hover:-rotate-6"
                    style={{ color: borderColor }}
                >
                    {icon}
                </div>
                <h3 
                    className="text-xl font-black mb-3 tracking-wide transition-colors"
                    style={{ color: 'var(--text-main)' }}
                >
                    {title}
                </h3>
                <p 
                    className="text-sm leading-relaxed transition-opacity"
                    style={{ color: 'var(--text-dim)' }}
                >
                    {description}
                </p>

                {/* Interactive Indicator */}
                <div 
                    className="mt-6 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.2em] transition-colors"
                    style={{ color: borderColor, opacity: 0.7 }}
                >
                    <span 
                        className="w-1.5 h-1.5 rounded-full animate-ping"
                        style={{ background: borderColor }}
                    />
                    System Active
                </div>
            </div>
        </motion.div>
    );
}
