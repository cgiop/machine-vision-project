import { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import './index.css';

const WS_URL   = 'ws://localhost:8000/ws/feed';
const API_URL  = 'http://localhost:8000';
const HOLD_MAX = 15;
const FRAME_MS = 120;

const TRANSLATE_LANGUAGES = [
  { code: 'fr', label: 'French',  flag: '🇫🇷' },
  { code: 'de', label: 'German',  flag: '🇩🇪' },
  { code: 'es', label: 'Spanish', flag: '🇪🇸' },
];

const MYMEMORY_URL = "https://api.mymemory.translated.net/get";

function drawBox(canvas, video, boxes, letter, conf) {
  if (!canvas || !video) return;
  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;
  canvas.width  = vw;
  canvas.height = vh;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, vw, vh);
  if (!boxes?.length) return;
  const col = '#fb923c';
  boxes.forEach(([x1, y1, x2, y2]) => {
    ctx.shadowColor = col; ctx.shadowBlur = 18;
    ctx.strokeStyle = col; ctx.lineWidth   = 2.5;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    ctx.shadowBlur  = 0;
    if (letter) {
      ctx.fillStyle = col;
      ctx.font      = 'bold 22px Inter, sans-serif';
      ctx.fillText(`${letter}  ${Math.round((conf||0)*100)}%`, x1+6, y1-8);
    }
  });
}

export default function App() {
  const [mode,        setMode]        = useState('alphabet');
  const [letter,      setLetter]      = useState('');
  const [conf,        setConf]        = useState(0);
  const [sentence,    setSentence]    = useState('');
  const [holdCount,   setHoldCount]   = useState(0);
  const [backend,     setBackend]     = useState(false);
  const [wsState,     setWsState]     = useState('connecting');
  const [fps,         setFps]         = useState(0);

  // Translation state
  const [transLang,   setTransLang]   = useState('fr');
  const [translated,  setTranslated]  = useState('');
  const [translating, setTranslating] = useState(false);
  const [transError,  setTransError]  = useState('');

  const webcamRef  = useRef(null);
  const canvasRef  = useRef(null);
  const wsRef      = useRef(null);
  const timerRef   = useRef(null);
  const fpsCount   = useRef(0);

  useEffect(() => {
    const id = setInterval(() => { setFps(fpsCount.current); fpsCount.current = 0; }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const check = () =>
      fetch(`${API_URL}/api/status`).then(r => setBackend(r.ok)).catch(() => setBackend(false));
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let dead = false;

    function connect() {
      if (dead) return;
      const sock = new WebSocket(`${WS_URL}?mode=${mode}`);
      wsRef.current = sock;
      setWsState('connecting');

      sock.onopen = () => {
        if (dead) { sock.close(); return; }
        setWsState('open');

        clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
          if (sock.readyState !== WebSocket.OPEN) return;
          const wc = webcamRef.current;
          if (!wc) return;
          const shot = wc.getScreenshot();
          if (shot) sock.send(shot);
        }, FRAME_MS);
      };

      sock.onmessage = (e) => {
        if (dead) return;
        fpsCount.current++;
        try {
          const data = JSON.parse(e.data);
          setHoldCount(data.hold_count || 0);
          if (data.prediction) {
            setLetter(data.prediction);
            setConf(data.confidence || 0);
          } else {
            setLetter(''); setConf(0);
          }
          if (data.mode === 'alphabet') setSentence(data.sentence || '');
          if (canvasRef.current && webcamRef.current?.video) {
            drawBox(canvasRef.current, webcamRef.current.video,
                    data.boxes, data.prediction, data.confidence);
          }
        } catch {}
      };

      sock.onclose = () => {
        setWsState('closed');
        clearInterval(timerRef.current);
        if (!dead) setTimeout(connect, 1500);
      };

      sock.onerror = () => sock.close();
    }

    connect();

    return () => {
      dead = true;
      clearInterval(timerRef.current);
      wsRef.current?.close();
      setLetter(''); setConf(0); setHoldCount(0);
    };
  }, [mode]);

  // ── Translation handler ────────────────────────────────────────────────────
  const handleTranslate = useCallback(async () => {
  const text = sentence.trim();
  if (!text) return;

  setTranslating(true);
  setTransError('');
  setTranslated('');

  try {
    const res = await fetch(
      `${MYMEMORY_URL}?q=${encodeURIComponent(text)}&langpair=en|${transLang}`
    );

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const json = await res.json();

    if (json.responseData?.translatedText) {
      setTranslated(json.responseData.translatedText);
    } else {
      throw new Error("No translation returned");
    }

  } catch (err) {
    setTransError(`Translation failed: ${err.message}`);
  } finally {
    setTranslating(false);
  }
}, [sentence, transLang]);

  const speakTranslated = () => {
    if (!translated.trim()) return;
    const langMap = { fr: 'fr-FR', de: 'de-DE', es: 'es-ES' };
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(translated.trim());
    u.lang = langMap[transLang] || transLang;
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
  };

  const clearSentence = () => {
    setSentence('');
    setTranslated('');
    setTransError('');
  };

  const speakSentence = () => {
    if (!sentence.trim()) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(sentence.trim());
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
  };

  const copyText = () => navigator.clipboard.writeText(sentence.trim());

  const holdPct = Math.round((holdCount / HOLD_MAX) * 100);

  const COL = mode === 'alphabet' ? '#fb923c' : mode === 'phrase' ? '#2dd4bf' : '#f59e0b';

  const s = {
    root: {
      minHeight:'100vh', background:'#0a0a0f',
      display:'flex', flexDirection:'column', alignItems:'center',
      fontFamily:"'Inter','Segoe UI',sans-serif", color:'#e2e8f0', padding:'24px 16px',
    },
    header: {
      width:'100%', maxWidth:'1120px',
      display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'24px',
    },
    logo: {
      fontSize:'22px', fontWeight:'900', letterSpacing:'-0.5px',
      background:'linear-gradient(135deg,#fb923c,#f43f5e)',
      WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent',
    },
    statusRow: { display:'flex', gap:'10px', alignItems:'center' },
    badge: (ok) => ({
      padding:'4px 11px', borderRadius:'20px', fontSize:'11px', fontWeight:'700',
      background: ok ? 'rgba(34,197,94,.1)' : 'rgba(239,68,68,.1)',
      border:`1px solid ${ok ? 'rgba(34,197,94,.3)' : 'rgba(239,68,68,.3)'}`,
      color: ok ? '#4ade80' : '#f87171',
    }),
    body: {
      width:'100%', maxWidth:'1120px',
      display:'grid', gridTemplateColumns:'1fr 360px', gap:'20px',
    },
    camWrap: {
      position:'relative', borderRadius:'16px', overflow:'hidden',
      background:'#111118', border:'1px solid rgba(255,255,255,.07)', aspectRatio:'4/3',
    },
    camEl: { width:'100%', height:'100%', objectFit:'cover', display:'block' },
    cvs: { position:'absolute', inset:0, pointerEvents:'none' },
    bigLetter: {
      position:'absolute', top:'14px', left:'50%', transform:'translateX(-50%)',
      fontSize:'90px', fontWeight:'900', lineHeight:1,
      color: COL, textShadow:`0 0 50px ${COL}88`,
      letterSpacing:'-3px', pointerEvents:'none',
    },
    holdBar:  { position:'absolute', bottom:0, left:0, right:0, height:'5px', background:'rgba(255,255,255,.06)' },
    holdFill: { height:'100%', width:`${holdPct}%`, transition:'width .1s linear',
                background:'linear-gradient(90deg,#fb923c,#fcd34d)' },
    holdTip:  {
      position:'absolute', bottom:'12px', left:'50%', transform:'translateX(-50%)',
      fontSize:'12px', fontWeight:'700', color:'rgba(251,146,60,.85)',
      background:'rgba(0,0,0,.55)', padding:'3px 12px', borderRadius:'12px', whiteSpace:'nowrap',
    },
    fpsBadge: {
      position:'absolute', top:'10px', right:'12px',
      fontSize:'11px', fontWeight:'700', color:'rgba(255,255,255,.25)',
      background:'rgba(0,0,0,.4)', padding:'2px 8px', borderRadius:'8px',
    },
    sidebar: { display:'flex', flexDirection:'column', gap:'14px' },
    card:  { background:'rgba(255,255,255,.03)', border:'1px solid rgba(255,255,255,.07)', borderRadius:'14px', padding:'18px' },
    cTitle:{ fontSize:'10px', fontWeight:'800', textTransform:'uppercase', letterSpacing:'1.2px',
             color:'rgba(255,255,255,.3)', marginBottom:'12px' },
    modeRow: { display:'flex', gap:'8px' },
    modeBtn: (active, col) => ({
      flex:1, padding:'9px 0', borderRadius:'9px', border:'none', cursor:'pointer',
      fontSize:'12px', fontWeight:'700', transition:'all .2s',
      background: active ? `${col}22` : 'rgba(255,255,255,.05)',
      color:       active ? col       : 'rgba(255,255,255,.35)',
      borderColor: active ? `${col}55` : 'transparent',
      borderWidth:'1px', borderStyle:'solid',
    }),
    grid: { display:'grid', gridTemplateColumns:'repeat(7,1fr)', gap:'5px' },
    lCell: (active) => ({
      aspectRatio:'1', display:'flex', alignItems:'center', justifyContent:'center',
      borderRadius:'8px', fontSize:'14px', fontWeight:'800', transition:'all .2s',
      background:  active ? `${COL}22` : 'rgba(255,255,255,.03)',
      border:      active ? `1px solid ${COL}66` : '1px solid rgba(255,255,255,.05)',
      color:       active ? COL : 'rgba(255,255,255,.18)',
      boxShadow:   active ? `0 0 16px ${COL}55` : 'none',
    }),
    sentBox: {
      minHeight:'72px', background:'rgba(0,0,0,.3)',
      border:'1px solid rgba(255,255,255,.08)', borderRadius:'10px',
      padding:'12px 14px', fontSize:'20px', fontWeight:'700',
      color:'#f1f5f9', letterSpacing:'0.04em', wordBreak:'break-all',
    },
    transBox: {
      minHeight:'56px', background:'rgba(0,0,0,.25)',
      border:'1px solid rgba(255,255,255,.06)', borderRadius:'10px',
      padding:'10px 14px', fontSize:'17px', fontWeight:'600',
      color:'#cbd5e1', letterSpacing:'0.03em', wordBreak:'break-all',
    },
    btnRow: { display:'flex', gap:'8px', marginTop:'10px' },
    btn: (r,g,b) => ({
      flex:1, padding:'9px 0', borderRadius:'9px',
      border:`1px solid rgba(${r},${g},${b},.25)`, cursor:'pointer',
      fontSize:'12px', fontWeight:'700', background:`rgba(${r},${g},${b},.1)`,
      color:`rgb(${r},${g},${b})`, transition:'all .2s',
    }),
    confBar:  { height:'4px', borderRadius:'4px', background:'rgba(255,255,255,.07)', overflow:'hidden', marginTop:'8px' },
    confFill: { height:'100%', borderRadius:'4px', transition:'width .3s',
                background:`linear-gradient(90deg,${COL},#f43f5e)`,
                width:`${Math.round(conf*100)}%` },
    pill: (active, col) => ({
      padding:'7px 14px', borderRadius:'8px', fontSize:'13px', fontWeight:'700',
      background: active ? `${col}22` : 'rgba(255,255,255,.04)',
      border:     active ? `1px solid ${col}55` : '1px solid rgba(255,255,255,.06)',
      color:      active ? col : 'rgba(255,255,255,.3)',
    }),
    // Translation-specific
    transRow: { display:'flex', gap:'8px', alignItems:'center', marginBottom:'10px' },
    langSelect: {
      flex:1, padding:'8px 10px', borderRadius:'9px',
      background:'rgba(255,255,255,.06)', border:'1px solid rgba(255,255,255,.12)',
      color:'#e2e8f0', fontSize:'13px', fontWeight:'600', cursor:'pointer',
      outline:'none', appearance:'none',
      backgroundImage:`url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23888' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")`,
      backgroundRepeat:'no-repeat', backgroundPosition:'right 10px center',
      paddingRight:'28px',
    },
    transBtn: (loading) => ({
      padding:'8px 16px', borderRadius:'9px', border:'1px solid rgba(139,92,246,.35)',
      background: loading ? 'rgba(139,92,246,.08)' : 'rgba(139,92,246,.15)',
      color: loading ? 'rgba(167,139,250,.5)' : '#a78bfa',
      fontSize:'12px', fontWeight:'700', cursor: loading ? 'not-allowed' : 'pointer',
      whiteSpace:'nowrap', transition:'all .2s',
    }),
    transLabel: {
      fontSize:'10px', fontWeight:'800', textTransform:'uppercase',
      letterSpacing:'1.2px', color:'rgba(167,139,250,.5)', marginBottom:'8px',
    },
    transErrorMsg: {
      fontSize:'11px', color:'#f87171', marginTop:'6px', padding:'6px 10px',
      background:'rgba(239,68,68,.08)', border:'1px solid rgba(239,68,68,.2)',
      borderRadius:'8px',
    },
    speakTransBtn: {
      marginTop:'8px', width:'100%', padding:'8px 0', borderRadius:'9px',
      border:'1px solid rgba(167,139,250,.25)', cursor:'pointer',
      fontSize:'12px', fontWeight:'700', background:'rgba(167,139,250,.08)',
      color:'#a78bfa', transition:'all .2s',
    },
  };

  const wsOk = wsState === 'open';
  const selectedLang = TRANSLATE_LANGUAGES.find(l => l.code === transLang);

  return (
    <div style={s.root}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.logo}>🤟 ASL Vision</div>
        <div style={s.statusRow}>
          <span style={s.badge(backend)}>{backend ? '● Backend' : '○ Backend Offline'}</span>
          <span style={s.badge(wsOk)}>{wsOk ? `● WS (${fps}fps)` : `○ WS ${wsState}`}</span>
        </div>
      </div>

      <div style={s.body}>
        {/* Camera panel */}
        <div style={s.camWrap}>
          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored={true}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.7}
            videoConstraints={{ width:640, height:480, facingMode:'user' }}
            style={s.camEl}
          />
          <canvas ref={canvasRef} style={s.cvs} />

          {letter && <div style={s.bigLetter}>{letter}</div>}

          {mode === 'alphabet' && (
            <>
              {holdCount > 0 && (
                <div style={s.holdTip}>Hold {letter} — {holdCount}/{HOLD_MAX}</div>
              )}
              <div style={s.holdBar}><div style={s.holdFill} /></div>
            </>
          )}

          <div style={s.fpsBadge}>{fps} fps</div>
        </div>

        {/* Sidebar */}
        <div style={s.sidebar}>

          {/* Mode buttons */}
          <div style={s.card}>
            <div style={s.cTitle}>Detection Mode</div>
            <div style={s.modeRow}>
              <button style={s.modeBtn(mode==='alphabet','#fb923c')} onClick={() => setMode('alphabet')}>🔤 A–Z</button>
              <button style={s.modeBtn(mode==='phrase','#2dd4bf')}   onClick={() => setMode('phrase')}>💬 Phrases</button>
              <button style={s.modeBtn(mode==='gesture','#f59e0b')}  onClick={() => setMode('gesture')}>🤙 Gestures</button>
            </div>
          </div>

          {/* A-Z alphabet grid */}
          {mode === 'alphabet' && (
            <div style={s.card}>
              <div style={s.cTitle}>ASL Alphabet — hold steady to commit</div>
              <div style={s.grid}>
                {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map(a => (
                  <div key={a} style={s.lCell(letter?.toUpperCase() === a)}>{a}</div>
                ))}
              </div>
              <div style={s.confBar}><div style={s.confFill} /></div>
              <div style={{marginTop:'5px',fontSize:'11px',color:'rgba(255,255,255,.22)'}}>
                Confidence: {Math.round(conf*100)}%
              </div>
            </div>
          )}

          {/* Phrase pills */}
          {mode === 'phrase' && (
            <div style={s.card}>
              <div style={s.cTitle}>Phrases (LSTM · 30-frame warmup)</div>
              <div style={{display:'flex',gap:'8px',flexWrap:'wrap'}}>
                {['hello','thanks','iloveyou'].map(p => (
                  <div key={p} style={s.pill(letter?.toLowerCase()===p,'#2dd4bf')}>{p}</div>
                ))}
              </div>
              <div style={s.confBar}><div style={s.confFill} /></div>
            </div>
          )}

          {/* Gesture pills */}
          {mode === 'gesture' && (
            <div style={s.card}>
              <div style={s.cTitle}>Gestures (TFLite · instant)</div>
              <div style={{display:'flex',gap:'8px',flexWrap:'wrap'}}>
                {['Open','Close','Pointer','OK'].map(g => (
                  <div key={g} style={s.pill(letter===g,'#f59e0b')}>{g}</div>
                ))}
              </div>
              <div style={s.confBar}><div style={s.confFill} /></div>
            </div>
          )}

          {/* Sentence / output box */}
          <div style={s.card}>
            <div style={s.cTitle}>{mode==='alphabet' ? 'Spelled Text' : 'Detected Output'}</div>
            <div style={s.sentBox}>
              {sentence || (
                <span style={{color:'rgba(255,255,255,.15)',fontSize:'14px',fontWeight:'400'}}>
                  {mode==='alphabet' ? 'Hold ASL letters to spell words…' : 'Signs appear here…'}
                </span>
              )}
            </div>
            <div style={s.btnRow}>
              <button style={s.btn(34,197,94)}   onClick={speakSentence}>🔊 Speak</button>
              <button style={s.btn(96,165,250)}   onClick={copyText}>📋 Copy</button>
              <button style={s.btn(248,113,113)}  onClick={clearSentence}>🗑 Clear</button>
            </div>
          </div>

          {/* ── Translation card (alphabet mode only) ── */}
          {mode === 'alphabet' && (
            <div style={s.card}>
              <div style={s.cTitle}>Translate Spelled Text</div>

              <div style={s.transRow}>
                <select
                  style={s.langSelect}
                  value={transLang}
                  onChange={e => { setTransLang(e.target.value); setTranslated(''); setTransError(''); }}
                >
                  {TRANSLATE_LANGUAGES.map(l => (
                    <option key={l.code} value={l.code}>{l.flag} {l.label}</option>
                  ))}
                </select>
                <button
                  style={s.transBtn(translating)}
                  onClick={handleTranslate}
                  disabled={translating || !sentence.trim()}
                >
                  {translating ? '⏳ …' : '🌐 Translate'}
                </button>
              </div>

              {(translated || transError || translating) && (
                <>
                  <div style={s.transLabel}>
                    {selectedLang?.flag} {selectedLang?.label} translation
                  </div>
                  <div style={s.transBox}>
                    {translating ? (
                      <span style={{color:'rgba(255,255,255,.2)',fontSize:'13px',fontWeight:'400'}}>Translating…</span>
                    ) : transError ? (
                      <span style={{color:'#f87171',fontSize:'13px',fontWeight:'400'}}>{transError}</span>
                    ) : (
                      translated
                    )}
                  </div>
                  {translated && !translating && (
                    <button style={s.speakTransBtn} onClick={speakTranslated}>
                      🔊 Speak in {selectedLang?.label}
                    </button>
                  )}
                </>
              )}

              {!translated && !transError && !translating && (
                <div style={{fontSize:'11px',color:'rgba(255,255,255,.2)',lineHeight:'1.6'}}>
                  Spell a word or phrase above, then hit Translate to convert it to {selectedLang?.label}.
                </div>
              )}
            </div>
          )}

          {/* Hint */}
          <div style={{...s.card, fontSize:'12px', color:'rgba(255,255,255,.28)', lineHeight:'1.7'}}>
            {mode==='alphabet' && <><strong style={{color:`${COL}bb`}}>Spelling:</strong> Show an ASL hand shape → hold it still until the orange bar fills → letter commits automatically. Backspace gesture removes last letter.</>}
            {mode==='phrase'   && <><strong style={{color:'#2dd4bf99'}}>LSTM:</strong> Needs 30 frames (~3s) to warm up. Recognises: hello · thanks · iloveyou</>}
            {mode==='gesture'  && <><strong style={{color:'#f59e0baa'}}>TFLite:</strong> Instant. Recognises: Open · Close · Pointer · OK hand shapes</>}
          </div>

        </div>
      </div>
    </div>
  );
}