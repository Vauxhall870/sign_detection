import { useEffect, useRef, useState } from 'react'
import { deletePrototype, getPrototypes, getTestClips } from '../api'
import { useRecognizerSocket } from '../useRecognizerSocket'
import { DEV_MODE } from '../devMode'

const COUNTDOWN_SECONDS = 3
const MIN_CLIPS = 3
const MAX_CLIPS = 8
const RECORD_TIMEOUT_MS = 10000   // give up if the backend never returns a clip

/**
 * Teach a new sign: record 3-8 short clips, give it a name, save.
 *
 * "Recording" here is not a video file. After the countdown we ask the backend to keep the
 * next 45-frame window of landmarks from the live frame stream, the very same window the
 * recogniser builds, so what we teach is what we later recognise. The clips stay on the
 * backend (tied to this connection) until "Save sign", so keep the camera on until then.
 */
export default function Teach({ health }) {
  const [name, setName] = useState('')
  const [clipCount, setClipCount] = useState(0)
  const [phase, setPhase] = useState('idle')        // idle -> countdown -> recording -> idle
  const [countdown, setCountdown] = useState(0)
  const [message, setMessage] = useState(null)      // { kind: 'ok' | 'error', text }
  const [clash, setClash] = useState(null)          // built-in sign the new one resembles
  const [taught, setTaught] = useState({ signs: [], threshold: null })
  const [live, setLive] = useState(null)            // best taught match for the current window
  // "Test clips" mode: record clips of a sign for later accuracy tests (session 2) WITHOUT teaching it
  const [testMode, setTestMode] = useState(false)
  const [testCounts, setTestCounts] = useState({ session1: {}, session2: {} })

  const phaseRef = useRef(phase)                    // so the socket callback sees the latest phase
  phaseRef.current = phase
  const timers = useRef([])                         // countdown interval + recording timeout

  const clearTimers = () => { timers.current.forEach((t) => { clearInterval(t); clearTimeout(t) }); timers.current = [] }
  const refreshTaught = () => getPrototypes().then(setTaught).catch(() => {})
  const refreshTestCounts = () => getTestClips().then(setTestCounts).catch(() => {})
  useEffect(() => { refreshTaught(); if (DEV_MODE) refreshTestCounts(); return clearTimers }, [])

  const { videoRef, canvasRef, streaming, status, start, stop, send } = useRecognizerSocket((msg) => {
    // The Teach tab only needs taught-sign matching, so it runs the backend in "isl" mode
    // (taught signs only: faster, and exactly what the live test panel shows).
    if (msg.type === 'ready') send({ action: 'mode', mode: 'isl' })
    if (msg.type === 'prediction') setLive(msg.taught)

    if (msg.type === 'clips') {
      setClipCount(msg.count)
      if (phaseRef.current === 'recording') {      // this reply is the clip we were waiting for
        clearTimers()
        setPhase('idle')
        setMessage({ kind: 'ok', text: `Clip ${msg.count} recorded.` })
      }
    }
    if (msg.type === 'clip_rejected') {
      clearTimers()
      setPhase('idle')
      setMessage({ kind: 'error', text: msg.reason })
    }
    if (msg.type === 'save_error') setMessage({ kind: 'error', text: msg.message })
    if (msg.type === 'saved_test') {
      setName('')
      setClipCount(0)
      setMessage({ kind: 'ok', text: `Saved ${msg.clips} test clips for "${msg.name}" (session 2 now has ${msg.total}). No sign was taught.` })
      refreshTestCounts()
    }
    if (msg.type === 'clash') setClash(msg)        // ask the user before saving
    if (msg.type === 'saved') {
      setClash(null)
      setName('')
      setClipCount(0)
      setMessage({ kind: 'ok', text: `Saved "${msg.name}" from ${msg.clips} clips.` })
      refreshTaught()
    }
  })

  // if the connection drops, the backend forgets our clips
  useEffect(() => {
    if (!streaming) { clearTimers(); setPhase('idle'); setClipCount(0); setClash(null); setLive(null) }
  }, [streaming])

  // 3-2-1 countdown, then "Sign now!" while the backend captures one window
  const record = () => {
    setMessage(null)
    setClash(null)
    setPhase('countdown')
    let n = COUNTDOWN_SECONDS
    setCountdown(n)
    const tick = setInterval(() => {
      n -= 1
      if (n > 0) { setCountdown(n); return }
      clearInterval(tick)
      setPhase('recording')
      send({ action: 'capture' })
      timers.current.push(setTimeout(() => {
        setPhase('idle')
        setMessage({ kind: 'error', text: 'No clip received. Is the camera running?' })
      }, RECORD_TIMEOUT_MS))
    }, 1000)
    timers.current.push(tick)
  }

  const save = (confirm = false) => {
    setMessage(null)                                 // drop stale messages such as "Clip 3 recorded."
    send({ action: 'save', name, confirm })
  }
  const saveTest = () => { setMessage(null); send({ action: 'save_test', name }) }
  const removeClip = (index) => send({ action: 'delete_clip', index })
  const removeSign = (signName) => deletePrototype(signName).then(refreshTaught)

  const canRecord = streaming && phase === 'idle' && clipCount < MAX_CLIPS
  const canSave = streaming && phase === 'idle' && name.trim() && clipCount >= (testMode ? 1 : MIN_CLIPS) && clipCount <= MAX_CLIPS
  const needsIslr = health && health.model !== 'islr'

  return (
    <div className="panel">
      {needsIslr && (
        <div className="note">Teaching needs the pretrained ISLR model (not the LSTM fallback).</div>
      )}

      <div className="stage">
        <video ref={videoRef} playsInline muted className="video" />
        <canvas ref={canvasRef} hidden />
        {!streaming && <div className="overlay">Camera off</div>}
        {phase === 'countdown' && <div className="cue countdown">{countdown}</div>}
        {phase === 'recording' && <div className="cue">Sign now!</div>}
      </div>

      <div className="controls">
        {streaming ? (
          <button className="danger" onClick={stop}>Stop camera</button>
        ) : (
          <button onClick={start} disabled={!health?.model_trained || needsIslr}>Start camera</button>
        )}
        <span className="status">{status}</span>
      </div>

      {/* Developer mode only (?dev=1): record clips for accuracy tests instead of teaching */}
      {DEV_MODE && (
        <label className="test-toggle">
          <input type="checkbox" checked={testMode} onChange={(e) => { setTestMode(e.target.checked); setClash(null); setMessage(null) }} />
          {' '}Record test clips only (session 2): saves clips for accuracy tests and does <strong>not</strong> teach or change any sign
        </label>
      )}

      <h3 className="section">1. Name the sign</h3>
      <div className="composer">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Sign name, e.g. namaste"
          maxLength={40}
          aria-label="Sign name"
          list={DEV_MODE ? 'session1-signs' : undefined}
        />
        {/* suggest the exact session-1 sign names so the two sessions line up */}
        {DEV_MODE && (
          <datalist id="session1-signs">
            {Object.keys(testCounts.session1).map((n) => <option key={n} value={n} />)}
          </datalist>
        )}
      </div>

      <h3 className="section">2. Record {MIN_CLIPS}-{MAX_CLIPS} clips</h3>
      <div className="controls">
        <button onClick={record} disabled={!canRecord}>Record clip</button>
        <span className="counter">Clips recorded: {clipCount} / {MAX_CLIPS} (need {testMode ? 1 : MIN_CLIPS}-{MAX_CLIPS})</span>
      </div>
      <div className="timeline">
        {Array.from({ length: clipCount }, (_, i) => (
          <button key={i} className="chip" onClick={() => removeClip(i)} title="Delete this clip" disabled={phase !== 'idle'}>
            Clip {i + 1} ✕
          </button>
        ))}
      </div>

      <h3 className="section">3. {testMode ? 'Save test clips' : 'Save'}</h3>
      <div className="controls">
        {testMode ? (
          <button onClick={saveTest} disabled={!canSave}>Save test clips</button>
        ) : (
          <button onClick={() => save(false)} disabled={!canSave}>Save sign</button>
        )}
      </div>
      {testMode && (
        <p className="muted">
          Session 2 clips so far: {Object.keys(testCounts.session1).map((n) => `${n} ${testCounts.session2[n] ?? 0}`).join(' · ') || 'none'}
        </p>
      )}
      {clash && (
        <div className="banner warn">
          {clash.clashes.map((c) => (
            <div key={c.kind + c.label}>
              {c.kind === 'builtin' && <>This looks like built-in ASL sign “{c.label}” ({(c.confidence * 100).toFixed(0)}% sure).</>}
              {c.kind === 'taught' && <>This looks like a sign you already taught, “{c.label}” (similarity {c.confidence.toFixed(2)}).</>}
              {c.kind === 'overwrite' && <>You already taught a sign called “{c.label}”; saving replaces it.</>}
            </div>
          ))}
          Save anyway?
          <span className="inline-actions">
            <button onClick={() => save(true)}>Save anyway</button>
            <button className="ghost" onClick={() => setClash(null)}>Cancel</button>
          </span>
        </div>
      )}
      {message && <div className={`banner ${message.kind === 'ok' ? 'ok' : 'error'}`}>{message.text}</div>}

      <h3 className="section">Live test</h3>
      <div className="live-test">
        {!streaming ? (
          <em>Start the camera, then sign to see the closest taught sign.</em>
        ) : taught.signs.length === 0 ? (
          <em>No signs taught yet.</em>
        ) : live ? (
          <>
            <div>
              Best taught match: <strong>{live.name}</strong>, similarity{' '}
              <strong>{live.similarity.toFixed(2)}</strong>
              {taught.threshold != null && <> (threshold {taught.threshold.toFixed(2)})</>}{' '}
              <span className={live.matched ? 'pass' : 'fail'}>{live.matched ? '✓ match' : 'below threshold'}</span>
            </div>
            {/* bar = similarity, tick = threshold */}
            <div className="meter">
              <div style={{ width: `${Math.max(0, Math.min(1, live.similarity)) * 100}%` }} />
              {taught.threshold != null && <i style={{ left: `${taught.threshold * 100}%` }} />}
            </div>
          </>
        ) : (
          <em>Sign something…</em>
        )}
      </div>

      <h3 className="section">Taught signs</h3>
      {taught.signs.length === 0 ? (
        <p className="muted">None yet.</p>
      ) : (
        <ul className="taught-list">
          {taught.signs.map((s) => (
            <li key={s.name}>
              <span><strong>{s.name}</strong> <span className="muted">({s.clips} clips)</span></span>
              <button className="ghost" onClick={() => removeSign(s.name)}>Delete</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
