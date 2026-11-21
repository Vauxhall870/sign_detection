export const API_BASE = 'http://localhost:8000'
export const WS_URL = 'ws://localhost:8000/ws/recognize'

export const mediaUrl = (src) => `${API_BASE}${src}`

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

// Signs taught so far, plus the similarity needed to match them: { signs: [{name, clips}], threshold }
export async function getPrototypes() {
  const res = await fetch(`${API_BASE}/api/prototypes`)
  if (!res.ok) throw new Error('Could not load taught signs')
  return res.json()
}

export async function deletePrototype(name) {
  const res = await fetch(`${API_BASE}/api/prototypes/${encodeURIComponent(name)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Could not delete sign')
}

export async function textToSign(text) {
  const res = await fetch(`${API_BASE}/api/text-to-sign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error('Translation failed')
  return res.json()
}
