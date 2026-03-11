import React from 'react';
import { motion } from 'framer-motion';

export default function ModalityCard({ title, icon, description, onClick, color = "primary" }) {
    const colorMap = {
        primary: "border-primary/30 group-hover:border-primary",
        secondary: "border-secondary/30 group-hover:border-secondary",
    };

    const glowMap = {
        primary: "shadow-[0_0_15px_rgba(34,211,238,0.2)] group-hover:shadow-[0_0_30px_rgba(34,211,238,0.5)]",
        secondary: "shadow-[0_0_15px_rgba(168,85,247,0.2)] group-hover:shadow-[0_0_30px_rgba(168,85,247,0.5)]",
    };

    return (
        <motion.div
            whileHover={{ scale: 1.05, rotateY: 5, rotateX: -5 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            onClick={onClick}
            className={`group relative glass-morphism p-8 rounded-3xl cursor-pointer transition-all duration-500 overflow-hidden ${colorMap[color]} ${glowMap[color]}`}
        >
            {/* Background Gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

            {/* Animated Glow Spot */}
            <div className="absolute -inset-2 bg-gradient-to-r from-primary/20 via-secondary/20 to-primary/20 opacity-0 group-hover:opacity-100 blur-2xl transition-opacity animate-pulse" />

            <div className="relative z-10 flex flex-col items-center text-center">
                <div className={`text-5xl mb-6 transition-transform group-hover:scale-110 group-hover:rotate-12`}>
                    {icon}
                </div>
                <h3 className="text-2xl font-black mb-3 tracking-wider text-white group-hover:text-primary transition-colors">
                    {title}
                </h3>
                <p className="text-dim text-sm leading-relaxed opacity-60 group-hover:opacity-100 transition-opacity">
                    {description}
                </p>

                {/* Interactive Indicator */}
                <div className="mt-6 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.2em] text-primary/50 group-hover:text-primary">
                    <span className="w-1.5 h-1.5 rounded-full bg-current animate-ping" />
                    Neural Link Active
                </div>
            </div>
        </motion.div>
    );
}
