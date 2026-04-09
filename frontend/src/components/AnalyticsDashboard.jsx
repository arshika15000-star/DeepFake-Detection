import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { ArrowLeft, RefreshCw, AlertTriangle, ShieldCheck, Activity } from 'lucide-react';

const API_BASE = "http://127.0.0.1:8005"; // fallback

export default function AnalyticsDashboard({ onBack, isDark }) {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchLogs = async () => {
    setLoading(true);
    let baseUrl = API_BASE;
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) {
      baseUrl = import.meta.env.VITE_API_BASE_URL;
    }
    try {
      const res = await axios.get(`${baseUrl}/logs`);
      setLogs(res.data);
      setError('');
    } catch (err) {
      console.error(err);
      setError('Failed to fetch analytics data. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const totalDetections = logs.length;
  const fakeCount = logs.filter(l => l.prediction === 'FAKE').length;
  const realCount = logs.filter(l => l.prediction === 'REAL').length;
  
  // Pie chart data
  const pieData = [
    { name: 'FAKE', value: fakeCount },
    { name: 'REAL', value: realCount }
  ];
  const COLORS = ['#fb7185', '#7ec8a0'];

  // Modality aggregation
  const modData = {};
  logs.forEach(l => {
    modData[l.modality] = (modData[l.modality] || 0) + 1;
  });
  const barData = Object.keys(modData).map(k => ({ name: k.toUpperCase(), value: modData[k] }));

  const bg = isDark ? '#1e293b' : '#ffffff';
  const textCol = isDark ? '#f8fafc' : '#0f172a';
  const subCol = isDark ? '#cbd5e1' : '#64748b';
  const cardBg = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)';

  return (
    <div className="w-full max-w-6xl mx-auto p-4 md:p-8 pt-24 min-h-[85vh] fade-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <button onClick={onBack} className="flex items-center gap-2 mb-2 text-sm font-semibold hover:opacity-80 transition-opacity" style={{ color: '#6390ff' }}>
            <ArrowLeft size={16} /> Back to Home
          </button>
          <h1 className="text-4xl font-bold" style={{ color: textCol }}>Analytics Dashboard</h1>
          <p style={{ color: subCol }} className="mt-1">Real-time usage and prediction metrics.</p>
        </div>
        <button onClick={fetchLogs} className="p-3 rounded-full hover:bg-white/10 transition-colors" title="Refresh">
          <RefreshCw size={20} style={{ color: textCol }} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-20">
          <Activity className="animate-pulse" size={40} style={{ color: '#7ec8a0' }} />
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl text-center" style={{ background: 'rgba(251,113,133,0.1)' }}>
          <p style={{ color: '#fb7185' }}>{error}</p>
        </div>
      ) : totalDetections === 0 ? (
        <div className="p-10 text-center rounded-2xl" style={{ background: cardBg }}>
          <p style={{ color: subCol }}>No analysis data found yet.</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Top Stat Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-2xl shadow-lg border" style={{ background: cardBg, borderColor: 'rgba(126,200,160,0.2)' }}>
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color: subCol }}>Total Scans</h3>
                <Activity size={20} style={{ color: '#6390ff' }} />
              </div>
              <div className="text-4xl font-black" style={{ color: textCol }}>{totalDetections}</div>
            </div>
            
            <div className="p-6 rounded-2xl shadow-lg border" style={{ background: cardBg, borderColor: 'rgba(251,113,133,0.2)' }}>
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color: subCol }}>Deepfakes Detected</h3>
                <AlertTriangle size={20} style={{ color: '#fb7185' }} />
              </div>
              <div className="text-4xl font-black" style={{ color: '#fb7185' }}>{fakeCount}</div>
            </div>

            <div className="p-6 rounded-2xl shadow-lg border" style={{ background: cardBg, borderColor: 'rgba(126,200,160,0.2)' }}>
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color: subCol }}>Authentic Media</h3>
                <ShieldCheck size={20} style={{ color: '#7ec8a0' }} />
              </div>
              <div className="text-4xl font-black" style={{ color: '#7ec8a0' }}>{realCount}</div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-6 rounded-2xl border" style={{ background: cardBg, borderColor: 'rgba(255,255,255,0.1)' }}>
              <h3 className="text-lg font-bold mb-6" style={{ color: textCol }}>Verdict Distribution</h3>
              <div className="w-full h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: bg, border: 'none', borderRadius: '8px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-4 text-sm mt-2 font-semibold">
                <span style={{ color: COLORS[0] }}>FAKE: {((fakeCount/totalDetections)*100).toFixed(1)}%</span>
                <span style={{ color: COLORS[1] }}>REAL: {((realCount/totalDetections)*100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="p-6 rounded-2xl border" style={{ background: cardBg, borderColor: 'rgba(255,255,255,0.1)' }}>
              <h3 className="text-lg font-bold mb-4" style={{ color: textCol }}>Scans by Modality</h3>
              <div className="w-full h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 20, right: 30, left: -20, bottom: 5 }}>
                    <XAxis dataKey="name" stroke={subCol} fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke={subCol} fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: bg, border: 'none', borderRadius: '8px' }} />
                    <Bar dataKey="value" fill="#6390ff" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* History Log */}
          <div className="p-6 rounded-2xl border overflow-hidden" style={{ background: cardBg, borderColor: 'rgba(255,255,255,0.1)' }}>
            <h3 className="text-lg font-bold mb-4" style={{ color: textCol }}>Recent Activity</h3>
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-left text-sm whitespace-nowrap">
                <thead>
                  <tr className="border-b" style={{ borderColor: 'rgba(255,255,255,0.1)', color: subCol }}>
                    <th className="py-3 px-4 font-semibold">Time</th>
                    <th className="py-3 px-4 font-semibold">Modality</th>
                    <th className="py-3 px-4 font-semibold">Prediction</th>
                    <th className="py-3 px-4 font-semibold">Confidence</th>
                    <th className="py-3 px-4 font-semibold">Job ID</th>
                  </tr>
                </thead>
                <tbody>
                  {[...logs].reverse().slice(0, 10).map((log, i) => (
                    <tr key={i} className="border-b last:border-0 hover:bg-neutral-500/5 transition-colors" style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
                      <td className="py-3 px-4" style={{ color: textCol }}>{new Date(log.timestamp).toLocaleString()}</td>
                      <td className="py-3 px-4 uppercase font-bold text-xs" style={{ color: '#c87eff' }}>{log.modality}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-md text-xs font-bold ${log.prediction === 'FAKE' ? 'bg-rose-500/10 text-rose-500' : 'bg-emerald-500/10 text-emerald-500'}`}>
                          {log.prediction}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono" style={{ color: textCol }}>{(log.confidence * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-mono text-xs" style={{ color: subCol }}>{log.job_id.split('-')[0]}...</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
