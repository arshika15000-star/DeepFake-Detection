import { useState, useRef, useEffect } from 'react';

// Custom SVG icon matching the uploaded image (cute white robot with teal eyes)
const RobotIcon = ({ size = 24, className }) => (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
        <rect x="20" y="35" width="60" height="45" rx="20" fill="white" />
        <rect x="25" y="40" width="50" height="35" rx="15" fill="#f0fdfa" />
        <circle cx="38" cy="55" r="8" fill="#0d9488" className="animate-pulse">
            <animate attributeName="r" values="7;9;7" dur="2s" repeatCount="indefinite" />
        </circle>
        <circle cx="62" cy="55" r="8" fill="#0d9488" className="animate-pulse">
            <animate attributeName="r" values="7;9;7" dur="2s" repeatCount="indefinite" />
        </circle>
        <path d="M45 70 Q50 72 55 70" stroke="#0d9488" strokeWidth="2" strokeLinecap="round" />
        <rect x="35" y="25" width="30" height="12" rx="6" fill="white" />
        <rect x="42" y="28" width="16" height="4" rx="2" fill="#0d9488" />
    </svg>
);

export default function Robot() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        {
            id: 1,
            type: 'bot',
            text: "Hello! I'm your Deepfake Detection Assistant. I can help analyze Image, Video, Audio, or Text for deepfakes. What would you like to check?",
            options: ['Image', 'Video', 'Audio', 'Text', 'Info']
        }
    ]);
    const [input, setInput] = useState('');
    const [mode, setMode] = useState('menu');
    const [fileType, setFileType] = useState(null);
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    const handleOptionClick = (option) => {
        const userMsg = { id: Date.now(), type: 'user', text: option };
        setMessages(prev => [...prev, userMsg]);

        if (option === 'Info') {
            setTimeout(() => {
                setMessages(prev => [...prev, {
                    id: Date.now() + 1,
                    type: 'bot',
                    text: "I use advanced AI models to detect anomalies in media files. \n\n• For Images/Videos: I look for unnatural artifacts and inconsistencies.\n• For Audio: I analyze distinct waveforms typical of synthetic voices.\n• For Text: I check for patterns common in AI-generated content.\n\nSelect a media type to start!",
                    options: ['Image', 'Video', 'Audio', 'Text', 'Info']
                }]);
            }, 500);
            return;
        }

        if (option === 'Text') {
            setMode('text_input');
            setTimeout(() => {
                setMessages(prev => [...prev, {
                    id: Date.now() + 1,
                    type: 'bot',
                    text: "Please type or paste the text you want me to analyze below."
                }]);
            }, 500);
        } else {
            setMode('file_upload');
            setFileType(option.toLowerCase());
            setTimeout(() => {
                setMessages(prev => [...prev, {
                    id: Date.now() + 1,
                    type: 'bot',
                    text: `Please upload the ${option} file you want to check.`,
                    isFileUpload: true,
                    accept: option === 'Video' ? 'video/*' : option === 'Audio' ? 'audio/*' : 'image/*'
                }]);
            }, 500);
        }
    };

    const handleTextSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { id: Date.now(), type: 'user', text: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const formData = new FormData();
            formData.append('current_text', input);

            const res = await fetch('http://localhost:8005/predict_text', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error('API Error');

            const data = await res.json();
            const botResponse = {
                id: Date.now() + 1,
                type: 'bot',
                text: `Analysis Complete:\nPrediction: ${data.prediction}\nConfidence: ${(data.confidence * 100).toFixed(1)}%\n(Real: ${(data.probabilities.real * 100).toFixed(1)}%, Fake: ${(data.probabilities.fake * 100).toFixed(1)}%)`
            };

            setMessages(prev => [...prev, botResponse, {
                id: Date.now() + 2,
                type: 'bot',
                text: "Would you like to check anything else?",
                options: ['Image', 'Video', 'Audio', 'Text', 'Info']
            }]);
            setMode('menu');

        } catch (err) {
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                type: 'bot',
                text: "Sorry, I encountered an error checking that text. Please try again."
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const userMsg = { id: Date.now(), type: 'user', text: `Uploaded ${file.name}` };
        setMessages(prev => [...prev, userMsg]);
        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        let endpoint = '';
        if (fileType === 'image') endpoint = '/predict_image';
        else if (fileType === 'video') endpoint = '/predict';
        else if (fileType === 'audio') endpoint = '/predict_audio';

        try {
            const res = await fetch(`http://localhost:8005${endpoint}`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error('API Error');

            const data = await res.json();
            const botResponse = {
                id: Date.now() + 1,
                type: 'bot',
                text: `Analysis Result:\nPrediction: ${data.prediction}\nConfidence: ${(data.confidence * 100).toFixed(1)}%\n(Real: ${(data.probabilities.real * 100).toFixed(1)}%, Fake: ${(data.probabilities.fake * 100).toFixed(1)}%)`
            };

            setMessages(prev => [...prev, botResponse, {
                id: Date.now() + 2,
                type: 'bot',
                text: "Would you like to check anything else?",
                options: ['Image', 'Video', 'Audio', 'Text', 'Info']
            }]);
            setMode('menu');

        } catch (err) {
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                type: 'bot',
                text: "Sorry, I had trouble analyzing that file. Please try again."
            }]);
        } finally {
            setLoading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const robotColor = '#0d9488'; // Teal-600
    const botBg = '#1e293b'; // Slate-800 for dark theme
    const userColor = '#6366f1'; // Indigo-500

    return (
        <>
            <button
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    position: 'fixed',
                    bottom: '24px',
                    right: '24px',
                    width: '70px',
                    height: '70px',
                    borderRadius: '24px',
                    backgroundColor: '#ffffff',
                    border: '2px solid #0d9488',
                    boxShadow: '0 10px 25px -5px rgba(13, 148, 136, 0.4)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 9999,
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                }}
                onMouseEnter={e => {
                    e.currentTarget.style.transform = 'translateY(-5px) scale(1.05)';
                    e.currentTarget.style.boxShadow = '0 20px 25px -5px rgba(13, 148, 136, 0.5)';
                }}
                onMouseLeave={e => {
                    e.currentTarget.style.transform = 'translateY(0) scale(1)';
                    e.currentTarget.style.boxShadow = '0 10px 25px -5px rgba(13, 148, 136, 0.4)';
                }}
            >
                {isOpen ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#0d9488" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                ) : (
                    <RobotIcon size={50} />
                )}
            </button>

            {isOpen && (
                <div style={{
                    position: 'fixed',
                    bottom: '110px',
                    right: '24px',
                    width: '400px',
                    height: '600px',
                    backgroundColor: '#0f172a',
                    borderRadius: '32px',
                    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
                    display: 'flex',
                    flexDirection: 'column',
                    zIndex: 9998,
                    overflow: 'hidden',
                    fontFamily: "'Inter', sans-serif",
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                    <div style={{
                        padding: '24px',
                        background: 'linear-gradient(135deg, #0d9488 0%, #0f766e 100%)',
                        color: 'white',
                        fontWeight: '700',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        fontSize: '1.1em'
                    }}>
                        <div style={{ backgroundColor: 'white', borderRadius: '12px', padding: '4px' }}>
                            <RobotIcon size={24} />
                        </div>
                        Assistant Bot
                    </div>

                    <div style={{
                        flex: 1,
                        padding: '24px',
                        overflowY: 'auto',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '16px',
                        background: '#0f172a'
                    }}>
                        {messages.map((msg) => (
                            <div key={msg.id} style={{
                                alignSelf: msg.type === 'user' ? 'flex-end' : 'flex-start',
                                maxWidth: '85%'
                            }}>
                                <div style={{
                                    padding: '14px 18px',
                                    borderRadius: '18px',
                                    borderBottomRightRadius: msg.type === 'user' ? '4px' : '18px',
                                    borderBottomLeftRadius: msg.type === 'bot' ? '4px' : '18px',
                                    backgroundColor: msg.type === 'user' ? '#a855f7' : '#1e293b',
                                    color: 'white',
                                    fontSize: '14px',
                                    lineHeight: '1.5',
                                    whiteSpace: 'pre-wrap',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}>
                                    {msg.text}
                                </div>

                                {msg.options && (
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '12px' }}>
                                        {msg.options.map(opt => (
                                            <button
                                                key={opt}
                                                onClick={() => handleOptionClick(opt)}
                                                style={{
                                                    padding: '8px 16px',
                                                    borderRadius: '12px',
                                                    border: '1px solid rgba(13, 148, 136, 0.5)',
                                                    backgroundColor: 'rgba(13, 148, 136, 0.1)',
                                                    color: '#2dd4bf',
                                                    fontSize: '13px',
                                                    cursor: 'pointer',
                                                    transition: 'all 0.2s',
                                                    fontWeight: '600'
                                                }}
                                                onMouseEnter={e => {
                                                    e.target.style.backgroundColor = '#0d9488';
                                                    e.target.style.color = 'white';
                                                }}
                                                onMouseLeave={e => {
                                                    e.target.style.backgroundColor = 'rgba(13, 148, 136, 0.1)';
                                                    e.target.style.color = '#2dd4bf';
                                                }}
                                            >
                                                {opt}
                                            </button>
                                        ))}
                                    </div>
                                )}

                                {msg.isFileUpload && (
                                    <div style={{ marginTop: '12px', backgroundColor: '#1e293b', padding: '12px', borderRadius: '12px', border: '1px dashed rgba(255,255,255,0.1)' }}>
                                        <input
                                            type="file"
                                            accept={msg.accept}
                                            onChange={handleFileUpload}
                                            style={{ color: '#94a3b8', fontSize: '12px' }}
                                            disabled={loading}
                                        />
                                    </div>
                                )}
                            </div>
                        ))}
                        {loading && (
                            <div style={{ alignSelf: 'flex-start', padding: '12px 18px', backgroundColor: '#1e293b', borderRadius: '18px', fontSize: '13px', color: '#94a3b8', borderBottomLeftRadius: '4px' }}>
                                <span className="animate-pulse">Analyzing media...</span>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div style={{
                        padding: '20px',
                        borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                        backgroundColor: '#0f172a'
                    }}>
                        <form onSubmit={handleTextSubmit} style={{ display: 'flex', gap: '12px' }}>
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Type deepfake query..."
                                disabled={mode !== 'text_input' || loading}
                                style={{
                                    flex: 1,
                                    padding: '12px 20px',
                                    borderRadius: '16px',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    backgroundColor: '#1e293b',
                                    color: 'white',
                                    fontSize: '14px',
                                    outline: 'none',
                                    transition: 'border-color 0.2s'
                                }}
                                onFocus={e => e.target.style.borderColor = '#0d9488'}
                                onBlur={e => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || loading || mode !== 'text_input'}
                                style={{
                                    backgroundColor: '#0d9488',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '16px',
                                    width: '48px',
                                    height: '48px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    cursor: (input.trim() && mode === 'text_input') ? 'pointer' : 'not-allowed',
                                    opacity: (input.trim() && mode === 'text_input') ? 1 : 0.5,
                                    boxShadow: '0 4px 6px -1px rgba(13, 148, 136, 0.4)'
                                }}
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                            </button>
                        </form>
                    </div>
                </div>
            )}
        </>
    );
}
