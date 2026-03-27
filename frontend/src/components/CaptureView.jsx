import React, { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Camera, X, Circle, Square, Mic, VideoOff } from 'lucide-react';

export default function CaptureView({ modality, onCapture, onClose }) {
    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [isRecording, setIsRecording] = useState(false);
    const [stream, setStream] = useState(null);
    const chunksRef = useRef([]);

    useEffect(() => {
        startStream();
        return () => stopAllTracks();
    }, [modality]);

    const stopAllTracks = () => {
        // Access current stream via ref to avoid stale closures
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
    };

    const startStream = async () => {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("Your browser does not support media devices. Make sure you are using localhost or HTTPS.");
            }

            const idealConstraints = {
                video: modality === 'image' || modality === 'video' ? { facingMode: 'user' } : false,
                audio: modality === 'audio' || modality === 'video' ? true : false,
            };

            let newStream;
            try {
                newStream = await navigator.mediaDevices.getUserMedia(idealConstraints);
            } catch (idealErr) {
                console.warn("Ideal constraints failed, trying basic fallback...", idealErr);
                try {
                    const basicConstraints = {
                        video: modality === 'image' || modality === 'video' ? true : false,
                        audio: modality === 'audio' || modality === 'video' ? true : false,
                    };
                    newStream = await navigator.mediaDevices.getUserMedia(basicConstraints);
                } catch (basicErr) {
                    console.warn("Basic combined fallback failed, attempting extreme isolation...", basicErr);
                    if (modality === 'video') {
                        newStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                        console.warn("Recovered by capturing Video-Only (No microphone detected).");
                    } else if (modality === 'audio') {
                        newStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    } else {
                        throw basicErr;
                    }
                }
            }

            setStream(newStream);
            if (videoRef.current) {
                videoRef.current.srcObject = newStream;
            }
        } catch (err) {
            console.error("Error accessing media devices:", err);
            let msg = "Unable to access camera or microphone.";
            if (err.name === 'NotAllowedError') msg = "Permission denied. Please click the 🔒 icon in your browser URL bar and allow Camera/Microphone access.";
            if (err.name === 'NotFoundError') msg = "No camera or microphone hardware found on this system.";

            alert(`Neural Interface Error: ${msg}\n\nTechnical details: ${err.message}`);
            onClose();
        }
    };

    const handleStopCamera = () => {
        // Stop any ongoing recording
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
        }
        stopAllTracks();
        onClose();
    };

    const captureImage = () => {
        if (!videoRef.current || videoRef.current.readyState < 2) {
            console.warn("Video stream not ready for capture");
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob((blob) => {
            if (!blob) {
                console.error("Failed to generate blob from canvas");
                return;
            }
            stopAllTracks();
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            onCapture(file);
        }, 'image/jpeg');
    };

    const startRecording = () => {
        if (!stream) {
            console.warn("No active stream to record");
            return;
        }

        chunksRef.current = [];

        let mimeType = '';
        if (modality === 'audio') {
            mimeType = ['audio/webm', 'audio/mp4', 'audio/wav'].find(type => MediaRecorder.isTypeSupported(type));
        } else {
            mimeType = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
        }

        const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                chunksRef.current.push(e.data);
            }
        };

        recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: recorder.mimeType || (modality === 'audio' ? 'audio/webm' : 'video/webm') });
            const filename = modality === 'audio' ? "capture.webm" : "capture.webm";
            const file = new File([blob], filename, { type: blob.type });
            stopAllTracks();
            onCapture(file);
        };

        recorder.start();
        setIsRecording(true);
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[100] backdrop-blur-3xl flex items-center justify-center p-6" style={{ background: 'var(--bg-card)' }}>
            <div className="relative w-full max-w-4xl glass-morphism rounded-[3rem] overflow-hidden shadow-2xl" style={{ borderColor: 'var(--border-subtle)' }}>
                {/* Header */}
                <div className="p-6 flex justify-between items-center border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                    <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--primary)' }} />
                        <span className="text-xs font-black uppercase tracking-[0.3em]" style={{ color: 'var(--primary)' }}>Live Neural Capture</span>
                    </div>
                    {/* Stop Camera Button */}
                    <button
                        onClick={handleStopCamera}
                        id="stop-camera-btn"
                        className="flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm transition-all hover:scale-105 active:scale-95"
                        style={{
                            background: 'rgba(251,113,133,0.15)',
                            border: '1px solid rgba(251,113,133,0.5)',
                            color: '#fb7185'
                        }}
                        title="Stop camera and close"
                    >
                        <VideoOff size={16} />
                        Stop Camera
                    </button>
                </div>

                {/* Viewport */}
                <div className="relative aspect-video flex items-center justify-center" style={{ background: '#000' }}>
                    {modality !== 'audio' ? (
                        <video
                            ref={videoRef}
                            autoPlay
                            muted
                            playsInline
                            className="w-full h-full object-cover"
                        />
                    ) : (
                        <div className="flex flex-col items-center gap-6">
                            <div className="w-32 h-32 rounded-full flex items-center justify-center relative shadow-lg" style={{ background: 'rgba(126,200,160,0.1)', borderColor: 'var(--primary)', borderWidth: '1px' }}>
                                <Mic size={48} style={{ color: 'var(--primary)' }} />
                                <motion.div
                                    animate={{ scale: [1, 1.2, 1] }}
                                    transition={{ repeat: Infinity, duration: 1.5 }}
                                    className="absolute inset-0 rounded-full border border-primary/30"
                                    style={{ borderColor: 'var(--primary)' }}
                                />
                            </div>
                            <span className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--text-dim)' }}>Audio Stream Active</span>
                        </div>
                    )}

                    {isRecording && (
                        <div className="absolute top-6 left-6 flex items-center gap-2 px-3 py-1 rounded-full" style={{ background: 'rgba(251,113,133,0.2)', borderColor: 'rgba(251,113,133,0.4)', borderWidth: '1px' }}>
                            <div className="w-2 h-2 rounded-full bg-danger animate-ping" />
                            <span className="text-[10px] font-black uppercase tracking-widest text-danger">Recording...</span>
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div className="p-12 flex flex-col items-center gap-8" style={{ background: 'var(--bg-card)' }}>
                    {modality === 'image' ? (
                        <div className="flex flex-col items-center gap-4">
                            <button
                                onClick={captureImage}
                                id="capture-image-btn"
                                className="w-24 h-24 rounded-full flex items-center justify-center text-dark hover:scale-110 transition-transform shadow-[0_0_40px_rgba(255,255,255,0.4)] border-4"
                                style={{ background: '#fff', borderColor: 'var(--border-subtle)', color: '#000' }}
                            >
                                <Camera size={38} />
                            </button>
                            <span className="text-sm font-black uppercase tracking-widest" style={{ color: 'var(--text-main)' }}>Snap & Analyze</span>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-4">
                            <button
                                onClick={isRecording ? stopRecording : startRecording}
                                id={isRecording ? 'stop-recording-btn' : 'start-recording-btn'}
                                className={`w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-2xl border-4 ${isRecording ? 'scale-90 border-danger/20' : 'hover:scale-110 border-white/20'}`}
                                style={{
                                    background: isRecording ? '#fff' : 'var(--danger)',
                                    color: isRecording ? 'var(--dark)' : '#fff',
                                    borderColor: isRecording ? 'var(--danger)' : 'var(--border-subtle)'
                                }}
                            >
                                {isRecording ? <Square size={38} color="var(--danger)" /> : <Circle size={38} fill="currentColor" />}
                            </button>
                            <span className={`text-sm font-black uppercase tracking-widest ${isRecording ? 'text-danger animate-pulse' : ''}`} style={{ color: isRecording ? 'var(--danger)' : 'var(--text-main)' }}>
                                {isRecording ? 'Stop Recording' : 'Start Feed Capture'}
                            </span>
                        </div>
                    )}

                    <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-dim)', opacity: 0.6 }}>
                        {modality === 'image' ? "Press to capture frame" : isRecording ? "Press to stop recording" : "Press to start stream capture"}
                    </p>
                </div>
            </div>
        </div>
    );
}
