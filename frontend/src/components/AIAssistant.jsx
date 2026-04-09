import React, { useState, useEffect, useRef } from 'react';
import { Bot, X, Send, Cpu, Sparkles, Activity } from 'lucide-react';

export default function AIAssistant({ isDark }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, sender: 'ai', text: "Initializing TruthLens Core... I am your AI Detection Assistant. How can I help you analyze media today?" }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMsg = { id: Date.now(), sender: 'user', text: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    // Simulated AI response logic
    setTimeout(() => {
      let responseText = "";
      const query = userMsg.text.toLowerCase();

      if (query.includes('video') || query.includes('mp4')) {
        responseText = "For video analysis, I extract evenly spaced frames and process them through our Time-Distributed ResNet backbone to detect facial flickering and temporal inconsistencies.";
      } else if (query.includes('audio') || query.includes('voice')) {
        responseText = "Audio analysis leverages our Wav2Vec 2.0 discriminator to detect synthetic phase distortions and robotic acoustic uniformity in cloned voices.";
      } else if (query.includes('text') || query.includes('chatgpt')) {
        responseText = "I analyze text using our RoBERTa transformer, examining token predictability (perplexity) and structural variance (burstiness) to detect LLM generation.";
      } else if (query.includes('image') || query.includes('photo')) {
        responseText = "Image forensics utilizes EfficientNet-V2 to scan for microscopic generative artifacts, GAN grids, and unnatural asymmetrical blending.";
      } else if (query.includes('hello') || query.includes('hi')) {
        responseText = "Greetings! To get started, you can drop any media file into the dashboard, or paste a URL for me to securely process and evaluate.";
      } else {
        responseText = "My neural pathways are focused specifically on media forensics. Please upload a file to the dashboard or ask me about our detection methodologies!";
      }

      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'ai', text: responseText }]);
      setIsTyping(false);
    }, 1500);
  };

  const bg = isDark ? '#1e293b' : '#ffffff';
  const textCol = isDark ? '#f8fafc' : '#0f172a';
  const subCol = isDark ? '#cbd5e1' : '#64748b';
  const accent = '#6390ff';

  return (
    <div className="fixed bottom-6 right-6 z-[100] flex flex-col items-end">
      {/* ── Chat Window ── */}
      <div 
        className={`mb-4 overflow-hidden flex flex-col transition-all duration-500 ease-in-out transform origin-bottom-right ${isOpen ? 'scale-100 opacity-100' : 'scale-0 opacity-0 pointer-events-none'}`}
        style={{ 
          width: '350px', 
          height: '500px', 
          background: isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.9)', 
          backdropFilter: 'blur(16px)',
          borderRadius: '24px',
          border: `1px solid ${isDark ? 'rgba(99, 144, 255, 0.3)' : 'rgba(99, 144, 255, 0.2)'}`,
          boxShadow: `0 10px 40px -10px ${isDark ? 'rgba(99, 144, 255, 0.3)' : 'rgba(0,0,0,0.1)'}`
        }}
      >
        {/* Header */}
        <div className="p-4 border-b flex justify-between items-center bg-gradient-to-r from-[#6390ff]/20 to-transparent" style={{ borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)' }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full flex items-center justify-center bg-[#6390ff]/20 relative">
              <Cpu size={16} color={accent} className="animate-pulse" />
              <div className="absolute top-0 right-0 w-2 h-2 rounded-full bg-[#7ec8a0]"></div>
            </div>
            <div>
              <h3 className="font-bold text-sm tracking-wide" style={{ color: textCol }}>TruthLens Oracle</h3>
              <p className="text-[10px] uppercase font-bold tracking-widest text-[#7ec8a0] flex items-center gap-1">
                <Activity size={10} /> Online
              </p>
            </div>
          </div>
          <button onClick={() => setIsOpen(false)} className="p-2 rounded-full hover:bg-black/10 transition">
            <X size={18} style={{ color: subCol }} />
          </button>
        </div>

        {/* Message Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div 
                className={`max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed shadow-sm ${msg.sender === 'user' ? 'rounded-tr-sm' : 'rounded-tl-sm'}`}
                style={{ 
                  background: msg.sender === 'user' ? accent : (isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0,0,0,0.03)'),
                  color: msg.sender === 'user' ? '#fff' : textCol,
                  border: msg.sender === 'ai' ? `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'}` : 'none'
                }}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="max-w-[85%] p-4 rounded-2xl rounded-tl-sm flex items-center gap-1" style={{ background: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0,0,0,0.03)' }}>
                <div className="w-1.5 h-1.5 bg-[#6390ff] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-1.5 h-1.5 bg-[#6390ff] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-1.5 h-1.5 bg-[#6390ff] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-3 border-t" style={{ borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)' }}>
          <div className="relative flex items-center group">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask the Oracle..." 
              className="w-full py-3 pl-4 pr-12 rounded-xl text-sm focus:outline-none transition-all duration-300"
              style={{ 
                background: isDark ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.04)', 
                color: textCol,
                border: `1px solid ${isDark ? 'rgba(99, 144, 255, 0.2)' : 'rgba(99, 144, 255, 0.1)'}`
              }}
            />
            <button 
              onClick={handleSend}
              disabled={!input.trim()}
              className="absolute right-2 p-2 rounded-lg transition-all disabled:opacity-30 disabled:scale-95 hover:scale-110"
              style={{ background: `${accent}20`, color: accent }}
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* ── Floating Glowing Orb Toggle ── */}
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-16 h-16 rounded-full flex items-center justify-center relative group transition-transform hover:scale-110"
        style={{ 
          background: `linear-gradient(135deg, ${accent}, #8b5cf6)`,
          boxShadow: `0 0 30px ${accent}66`
        }}
      >
        <div className="absolute inset-0 rounded-full animate-ping opacity-20" style={{ background: accent }}></div>
        {isOpen ? <X size={28} color="#fff" /> : <Bot size={28} color="#fff" className="group-hover:animate-bounce" />}
        {!isOpen && (
          <div className="absolute -top-1 -right-1 w-4 h-4 bg-[#fb7185] rounded-full border-2 shadow-lg" style={{ borderColor: bg }}></div>
        )}
      </button>
    </div>
  );
}
