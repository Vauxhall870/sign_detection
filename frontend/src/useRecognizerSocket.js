import { useEffect, useRef, useState } from 'react'
import { WS_URL } from './api'

const SEND_FPS = 12          // frames/sec pushed to the backend
const JPEG_QUALITY = 0.6

/**
 * Camera + WebSocket to /ws/recognize, shared by the Sign → Text and Teach tabs.
 *
 * Both tabs push webcam frames through the SAME pipeline (same canvas size, mirroring and
 * JPEG settings, same backend frame buffer), so a sign recorded in the Teach tab looks to the
 * model exactly like a live sign later on.
 *
 * `onMessage(msg)` is called for every JSON message from the server. Returns refs for the
 * <video>/<canvas> elements, the connection state, and start / stop / send(object).
 */
export function useRecognizerSocket(onMessage) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const timerRef = useRef(null)
  // keep the latest callback without reconnecting when the component re-renders
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const [streaming, setStreaming] = useState(false)
  const [status, setStatus] = useState('idle')

  const stop = () => {
    clearInterval(timerRef.current)
    timerRef.current = null
    wsRef.current?.close()
    wsRef.current = null
    const tracks = videoRef.current?.srcObject?.getTracks() ?? []
    tracks.forEach((t) => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    setStreaming(false)
    setStatus('idle')
  }

  useEffect(() => stop, [])

  // send a command (e.g. {action: 'capture'}) to the backend
  const send = (obj) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj))
  }

  const sendFrame = () => {
    const ws = wsRef.current
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN || !video?.videoWidth) return

    // Downscale before encoding: MediaPipe does not need full resolution and
    // this keeps each frame well under a few hundred KB on the wire.
    canvas.width = 480
    canvas.height = (video.videoHeight / video.videoWidth) * 480
    const ctx = canvas.getContext('2d')
    ctx.save()
    ctx.scale(-1, 1)                                  // mirror, matches training
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
    ctx.restore()
    ws.send(JSON.stringify({ frame: canvas.toDataURL('image/jpeg', JPEG_QUALITY) }))
  }

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
    } catch {
      setStatus('camera denied')
      return
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    setStatus('connecting')

    ws.onopen = () => {
      setStreaming(true)
      timerRef.current = setInterval(sendFrame, 1000 / SEND_FPS)
    }

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'ready') setStatus('watching')
      if (msg.type === 'error') { setStatus(msg.message); stop() }
      onMessageRef.current?.(msg)
    }

    ws.onerror = () => setStatus('connection failed')
    ws.onclose = () => setStreaming(false)
  }

  return { videoRef, canvasRef, streaming, status, start, stop, send }
}
