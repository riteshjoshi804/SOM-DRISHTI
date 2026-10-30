// Navigation helper for the hash router (kept separate from App.jsx to avoid import cycles).
export function go(path) {
  window.location.hash = path
}
