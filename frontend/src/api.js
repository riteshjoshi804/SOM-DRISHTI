// Thin client for the FastAPI backend. Every value shown in the UI comes from here.

async function request(path, options) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail
    try {
      detail = (await res.json()).detail
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`)
  }
  return res.json()
}

const enc = encodeURIComponent

export const api = {
  health: () => request('/health'),
  dataset: () => request('/api/dataset'),
  scenes: () => request('/api/scenes'),
  scene: (id) => request(`/api/scenes/${enc(id)}`),
  previewUrl: (id, which) => `/api/scenes/${enc(id)}/preview/${which}`,
  register: (id) =>
    request(`/api/register/${enc(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    }),
  upload: (formData) => request('/api/register/upload', { method: 'POST', body: formData }),
  results: () => request('/api/results'),
  result: (jobId) => request(`/api/results/${enc(jobId)}`),
  metrics: (jobId) => request(`/api/results/${enc(jobId)}/metrics`),
  artifactUrl: (jobId, name) => `/api/results/${enc(jobId)}/artifacts/${enc(name)}`,
  defaultConfig: () => request('/api/config/default'),
}
