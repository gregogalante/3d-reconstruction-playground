// Talking to the capture server. Uploads go through a queue because the shutter must
// never wait for the network: on a phone at the far end of a house, a frame can take a
// second to land, and the next shot is due before that.

export async function listDatasets () {
  const response = await window.fetch('/api/capture/datasets')
  if (!response.ok) throw new Error('Could not read the datasets')
  return (await response.json()).datasets
}

export async function openSession (body) {
  const response = await window.fetch('/api/capture/sessions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  })
  if (!response.ok) throw new Error((await response.json()).detail || 'Could not open the capture')
  return response.json()
}

export async function finishSession (id) {
  const response = await window.fetch(`/api/capture/sessions/${id}/finish`, { method: 'POST' })
  if (!response.ok) throw new Error('Could not close the capture')
  return response.json()
}

export async function undoLast (id) {
  const response = await window.fetch(`/api/capture/sessions/${id}/frames/last`, { method: 'DELETE' })
  if (!response.ok) throw new Error('Nothing to undo')
  return response.json()
}

export function createUploader (id, { onChange = () => {} } = {}) {
  const queue = []
  let running = false
  let uploaded = 0
  let failed = 0

  const report = () => onChange({ pending: queue.length, uploaded, failed })

  async function send (item) {
    const form = new window.FormData()
    form.append('frame', item.blob, 'frame.jpg')
    form.append('meta', JSON.stringify(item.meta))
    const response = await window.fetch(`/api/capture/sessions/${id}/frames`, { method: 'POST', body: form })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  async function pump () {
    if (running) return
    running = true
    while (queue.length) {
      const item = queue[0]
      try {
        await send(item)
        uploaded++
      } catch {
        // One retry, then the frame is dropped and counted: a capture that stalls on a
        // bad frame is worse than a capture with a hole in it, and the count is shown.
        if (!item.retried) {
          item.retried = true
          report()
          continue
        }
        failed++
      }
      queue.shift()
      report()
    }
    running = false
  }

  return {
    push (blob, meta) {
      queue.push({ blob, meta })
      report()
      pump()
    },
    async drain () {
      while (queue.length || running) await new Promise(resolve => window.setTimeout(resolve, 120))
    },
    get counts () {
      return { pending: queue.length, uploaded, failed }
    }
  }
}
