import { useEffect, useState } from 'react'
import Dashboard from './pages/Dashboard.jsx'
import Scenes from './pages/Scenes.jsx'
import SceneDetail from './pages/SceneDetail.jsx'
import Results from './pages/Results.jsx'
import ResultDetail from './pages/ResultDetail.jsx'
import Methodology from './pages/Methodology.jsx'

// Hash routing keeps deep links working whether the app is served by Vite or by FastAPI.
function useRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  const [route, setRoute] = useState(read)
  useEffect(() => {
    const onChange = () => {
      setRoute(read())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

const NAV = [
  ['', 'Dashboard'],
  ['scenes', 'Dataset / Scenes'],
  ['results', 'Registration Results'],
  ['methodology', 'Methodology'],
]

export default function App() {
  const [section, id] = useRoute()
  let page
  if (section === 'scenes' && id) page = <SceneDetail sceneId={id} />
  else if (section === 'scenes') page = <Scenes />
  else if (section === 'results' && id) page = <ResultDetail jobId={id} />
  else if (section === 'results') page = <Results />
  else if (section === 'methodology') page = <Methodology />
  else page = <Dashboard />

  return (
    <div className="min-h-screen">
      <header className="border-b border-white/10 bg-surface/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <a href="#/" className="group">
            <div className="text-xl font-bold tracking-[0.18em]">SOM DRISHTI</div>
            <div className="text-xs text-ink-2">Lunar Image Correspondence &amp; Registration System</div>
          </a>
          <nav className="flex flex-wrap gap-1 text-sm">
            {NAV.map(([path, label]) => {
              const active = (section || '') === path
              return (
                <a
                  key={path}
                  href={`#/${path}`}
                  className={`rounded-lg px-3 py-1.5 transition-colors ${
                    active ? 'bg-white/10 text-ink' : 'text-ink-2 hover:bg-white/5'
                  }`}
                >
                  {label}
                </a>
              )
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 pb-16 pt-6 sm:px-6">{page}</main>
      <footer className="border-t border-white/10 py-6 text-center text-xs text-muted">
        SIH 2026 · Problem statement SIH26166 · ISRO / Department of Space · Classical SIFT MVP
      </footer>
    </div>
  )
}
