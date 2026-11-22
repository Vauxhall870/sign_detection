// Developer mode: add ?dev=1 to the URL (e.g. http://localhost:5173/?dev=1).
// It reveals testing-only controls (such as recording session-2 test clips). Normal mode, the
// default, shows only what a demo needs. Nothing is removed: the code and data stay in place.
export const DEV_MODE = new URLSearchParams(window.location.search).get('dev') === '1'
