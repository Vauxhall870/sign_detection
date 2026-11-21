import { useEffect, useRef, useState } from 'react'
import { getPrototypes } from '../api'
import { useRecognizerSocket } from '../useRecognizerSocket'

// How to print a result's confidence. The two sources use different scales:
// the built-in model gives a probability (0-100%), a taught sign gives a cosine similarity (0-1).
const confidenceText = (source, c) =>
  source === 'taught' ? `similarity ${c.toFixed(2)}` : `${(c * 100).toFixed(0)}%`

export default function SignToText({ health }) {
  const [prediction, setPrediction] = useState(null)
  const [sentence, setSentence] = useState([])
  const [buffering, setBuffering] = useState({ filled: 0, needed: 30 })

  // "asl" = built-in 250 signs + taught signs; "isl" = taught signs only
  const [mode, setMode] = useState('asl')
  const modeRef = useRef(mode)            // lets the socket callback read the latest mode
  modeRef.current = mode
  const [taught, setTaught] = useState({ signs: [], threshold: null })

  const refreshTaught = () => getPrototypes().then(setTaught).catch(() => {})
  useEffect(() => { refreshTaught() }, [])

  const { videoRef, canvasRef, streaming, status, start, stop, send } = useRecognizerSocket((msg) => {
    // as soon as the server is ready, tell it which mode we are in
    if (msg.type === 'ready') send({ action: 'mode', mode: modeRef.current })
    if (msg.type === 'buffering') setBuffering({ filled: msg.filled, needed: msg.needed })
    if (msg.type === 'prediction') {
      setBuffering({ filled: msg.scores ? 30 : 0, needed: 30 })
      setPrediction(msg)
      if (msg.accepted) setSentence((s) => [...s, msg.label])
    }
  })

  const chooseMode = (next) => {
    setMode(next)
    setPrediction(null)                    // old bars/result belong to the other mode
    send({ action: 'mode', mode: next })   // takes effect immediately, no reconnect
    refreshTaught()                        // pick up signs taught since this tab was opened
  }

  // built-in score bars (ASL mode only; ISL mode sends no built-in scores)
  const ranked = prediction
    ? Object.entries(prediction.scores).sort((a, b) => b[1] - a[1]).slice(0, 4)
    : []
  const nothingTaught = taught.signs.length === 0

  return (
    <div className="panel">
      <div className="modebar" role="group" aria-label="Recognition mode">
        <button className={mode === 'asl' ? 'active' : ''} onClick={() => chooseMode('asl')}>
          ASL mode
        </button>
        <button className={mode === 'isl' ? 'active' : ''} onClick={() => chooseMode('isl')}>
          ISL mode
        </button>
      </div>
      <p className="mode-help">
        {mode === 'asl'
          ? 'The built-in 250 ASL signs only. Taught signs are not used here.'
          : 'Only the signs you taught. Built-in results are ignored.'}
      </p>
      {mode === 'isl' && nothingTaught && (
        <div className="note">No signs taught yet. Open the Teach tab and teach some signs first.</div>
      )}

      <div className="stage">
        <video ref={videoRef} playsInline muted className="video" />
        <canvas ref={canvasRef} hidden />
        {!streaming && <div className="overlay">Camera off</div>}
        {prediction?.hands === false && streaming && (
          <div className="hint">No hands detected — move into frame</div>
        )}
      </div>

      <div className="controls">
        {streaming ? (
          <button className="danger" onClick={stop}>Stop</button>
        ) : (
          <button onClick={start} disabled={!health?.model_trained}>Start camera</button>
        )}
        <button className="ghost" onClick={() => setSentence([])}>Clear</button>
        <span className="status">{status}</span>
      </div>

      {streaming && buffering.filled < buffering.needed && (
        <div className="progress">
          <div style={{ width: `${(buffering.filled / buffering.needed) * 100}%` }} />
          <span>collecting frames {buffering.filled}/{buffering.needed}</span>
        </div>
      )}

      {/* current result: which sign, where it came from, how confident */}
      {prediction && (
        <div className="result">
          {prediction.source ? (
            <>
              <span className={`badge ${prediction.source}`}>{prediction.source}</span>
              <strong>{prediction.label}</strong>
              <span className="conf">{confidenceText(prediction.source, prediction.confidence)}</span>
            </>
          ) : (
            <em>Not sure</em>
          )}
        </div>
      )}

      {/* ISL mode: show the closest taught sign even when it is below the threshold */}
      {prediction?.mode === 'isl' && prediction.taught && (
        <p className="closest">
          Closest taught sign: <strong>{prediction.taught.name}</strong> ({prediction.taught.similarity.toFixed(2)}
          {taught.threshold != null && `, needs ${taught.threshold.toFixed(2)}`})
        </p>
      )}

      {prediction && ranked.length > 0 && (
        <div className="scores">
          {ranked.map(([label, score]) => (
            <div key={label} className="score-row">
              <span className="label">{label}</span>
              <div className="bar"><div style={{ width: `${score * 100}%` }} /></div>
              <span className="pct">{(score * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="sentence">
        <h3>Translation</h3>
        <p>{sentence.length ? sentence.join(' ') : <em>Sign something…</em>}</p>
      </div>
    </div>
  )
}
