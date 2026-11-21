// Browser end-to-end test for the Teach tab and the ASL / ISL modes: the real React app in
// Chrome, a recorded clip played through Chrome's fake webcam, the real FastAPI backend
// (with a throw-away prototypes file, so your real taught signs are never touched).
//
//   npm run test:teach
//
// Needs Google Chrome, the backend venv, data/raw (ml.fetch_wlasl) and the pretrained model.
import { spawn, execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const FRONTEND = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const BACKEND = resolve(FRONTEND, '../backend')
const PYTHON = [join(BACKEND, '.venv/Scripts/python.exe'), join(BACKEND, '.venv/bin/python')].find(existsSync)
const VITE = join(FRONTEND, 'node_modules/vite/bin/vite.js')
const APP = 'http://localhost:5173'
const API = 'http://localhost:8000'

const failures = []
function check(name, ok, detail = '') {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures.push(name)
}

const children = []
function start(cmd, args, cwd, env = {}) {
  const child = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, stdio: 'ignore' })
  children.push(child)
}
async function waitFor(url, ms = 120000) {
  const end = Date.now() + ms
  while (Date.now() < end) {
    try { if ((await fetch(url)).ok) return } catch {}
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`${url} did not come up`)
}

async function openApp(camera) {
  const browser = await chromium.launch({
    channel: 'chrome', headless: true,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${camera}`],
  })
  const page = await browser.newPage()
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(String(e)))
  await page.goto(APP)
  await page.waitForFunction(() => !document.querySelector('.controls button')?.disabled, null, { timeout: 20000 })
  return { browser, page, errors }
}
const text = (page, sel) => page.locator(sel).first().textContent()
const waitWatching = (page) =>
  page.waitForFunction(() => document.querySelector('.status')?.textContent === 'watching', null, { timeout: 20000 })
const counter = async (page) => Number((await text(page, '.counter')).match(/recorded: (\d)/)[1])

// Press "Record clip" until the counter reaches `target`. A clip is rejected when no hands were
// in view during the window (the fake webcam clip has still moments), exactly as for a real
// user who just tries again.
async function recordUntil(page, target, sawCue) {
  for (let attempt = 0; attempt < 30 && (await counter(page)) < target; attempt++) {
    await page.getByRole('button', { name: 'Record clip' }).click()
    // the cue overlay is on screen for the whole countdown + recording, then disappears
    const countdown = await page.locator('.cue.countdown').waitFor({ timeout: 3000 }).then(() => true, () => false)
    const signNow = await page.getByText('Sign now!').waitFor({ timeout: 8000 }).then(() => true, () => false)
    if (sawCue) { sawCue.countdown ||= countdown; sawCue.signNow ||= signNow }
    await page.locator('.cue').waitFor({ state: 'detached', timeout: 20000 })
  }
}

// ------------------------------------------------------------------ tests

async function islEmptyState(camera) {
  console.log('\nISL mode with nothing taught')
  const { browser, page, errors } = await openApp(camera)
  check('default mode is ASL', (await page.locator('.modebar button.active').textContent()) === 'ASL mode')
  await page.getByRole('button', { name: 'ISL mode', exact: true }).click()
  const note = await page.locator('.note').textContent()
  check('tells the user to teach signs first', /teach some signs first/i.test(note), note)
  await page.getByRole('button', { name: 'ASL mode', exact: true }).click()
  check('note disappears in ASL mode', (await page.locator('.note').count()) === 0)
  check('no console errors', errors.length === 0, errors.join(' | '))
  await browser.close()
}

async function teachFlow(camera) {
  console.log('\nTeach tab (fake webcam: a "drink" clip, a built-in sign)')
  const { browser, page, errors } = await openApp(camera)
  await page.getByRole('button', { name: 'Teach', exact: true }).click()

  check('Record and Save are disabled with the camera off',
    (await page.getByRole('button', { name: 'Record clip' }).isDisabled()) && (await page.getByRole('button', { name: 'Save sign' }).isDisabled()))
  check('taught list starts empty', (await text(page, '.taught-list, .muted')).includes('None yet'))

  await page.getByRole('button', { name: 'Start camera' }).click()
  await waitWatching(page)
  check('live test says nothing is taught yet', (await text(page, '.live-test')).includes('No signs taught yet'))
  await page.getByLabel('Sign name').fill('my drink')

  const cue = {}
  await recordUntil(page, 3, cue)
  check('3-2-1 countdown is shown', cue.countdown)
  check('"Sign now!" cue is shown', cue.signNow)
  check('counter reaches 3 / 8', (await counter(page)) === 3, await text(page, '.counter'))
  check('one chip per clip', (await page.locator('.timeline .chip').count()) === 3)
  check('Save is enabled with 3 clips and a name', await page.getByRole('button', { name: 'Save sign' }).isEnabled())

  await page.locator('.timeline .chip').first().click()
  await page.waitForFunction(() => /recorded: 2/.test(document.querySelector('.counter')?.textContent ?? ''), null, { timeout: 5000 })
  check('a bad clip can be deleted', (await counter(page)) === 2)
  check('Save is disabled again with 2 clips', await page.getByRole('button', { name: 'Save sign' }).isDisabled())
  await recordUntil(page, 3)
  check('re-recorded back to 3 clips', (await counter(page)) === 3)

  await recordUntil(page, 8)
  check('can record up to 8 clips', (await counter(page)) === 8, await text(page, '.counter'))
  check('Record is disabled at the 8-clip cap', await page.getByRole('button', { name: 'Record clip' }).isDisabled())

  // the clip is a built-in sign, so saving must warn first
  await page.getByRole('button', { name: 'Save sign' }).click()
  await page.locator('.banner.warn').waitFor({ timeout: 10000 })
  const warning = await text(page, '.banner.warn')
  check('clash warning names the built-in sign', /built-in ASL sign .drink/i.test(warning) && /Save anyway\?/.test(warning), warning.replace(/\s+/g, ' '))
  check('nothing was saved yet', (await (await fetch(`${API}/api/prototypes`)).json()).signs.length === 0)

  await page.getByRole('button', { name: 'Cancel' }).click()
  check('Cancel dismisses the warning and saves nothing',
    (await page.locator('.banner.warn').count()) === 0 && (await (await fetch(`${API}/api/prototypes`)).json()).signs.length === 0)

  await page.getByRole('button', { name: 'Save sign' }).click()
  await page.locator('.banner.warn').waitFor({ timeout: 10000 })
  await page.getByRole('button', { name: 'Save anyway' }).click()
  const saved = await page.waitForFunction(() => /Saved/.test(document.querySelector('.banner.ok')?.textContent ?? ''), null, { timeout: 10000 })
    .then(() => true, () => false)
  check('Save anyway saves the sign', saved && (await text(page, '.banner.ok')).includes('my drink'), await text(page, '.banner.ok').catch(() => ''))
  const evalFiles = existsSync(join(tmp, 'eval', 'my_drink')) ? readdirSync(join(tmp, 'eval', 'my_drink')) : []
  check('each clip is also saved for later accuracy tests', evalFiles.length === 8 && evalFiles.every((f) => f.endsWith('.npy')), evalFiles.join(', '))
  check('sign appears in the taught list', (await text(page, '.taught-list')).includes('my drink'))
  check('clip counter resets and the name box clears',
    (await counter(page)) === 0 && (await page.getByLabel('Sign name').inputValue()) === '')

  // the clip keeps looping, so the live panel should now show a close taught match
  await page.waitForFunction(() => /similarity/.test(document.querySelector('.live-test')?.textContent ?? ''), null, { timeout: 20000 })
  const live = await text(page, '.live-test')
  check('live test shows best match, similarity and threshold', /my drink/.test(live) && /similarity\s*\d\.\d\d/.test(live) && /threshold 0\.\d\d/.test(live), live.replace(/\s+/g, ' '))
  check('no console errors', errors.length === 0, errors.join(' | '))
  await browser.close()
}

async function modes(camera) {
  console.log('\nSign -> Text modes with "my drink" taught (webcam shows the built-in sign "drink")')
  const { browser, page, errors } = await openApp(camera)

  // ISL: taught signs only, so the built-in "drink" must not appear and the taught sign should
  await page.getByRole('button', { name: 'ISL mode', exact: true }).click()
  check('ISL: no "teach first" note once a sign is taught', (await page.locator('.note').count()) === 0)
  await page.getByRole('button', { name: 'Start camera' }).click()
  await waitWatching(page)
  const taught = await page.waitForFunction(() => document.querySelector('.badge.taught') !== null, null, { timeout: 30000 }).then(() => true, () => false)
  check('ISL: result is labelled "taught"', taught, taught ? await text(page, '.result') : '')
  check('ISL: taught sign goes into the sentence', (await text(page, '.sentence p')).includes('my drink'), await text(page, '.sentence p'))
  check('ISL: built-in score bars are hidden', (await page.locator('.score-row').count()) === 0)
  check('ISL: closest taught sign and its similarity are shown', /Closest taught sign: my drink \(\d\.\d\d/.test(await text(page, '.closest')), await text(page, '.closest'))

  // ASL: built-in is confident about "drink", so it wins over the taught sign
  await page.getByRole('button', { name: 'Clear' }).click()
  await page.getByRole('button', { name: 'ASL mode', exact: true }).click()
  const builtin = await page.waitForFunction(() => document.querySelector('.badge.built-in') !== null, null, { timeout: 30000 }).then(() => true, () => false)
  check('ASL: result is labelled "built-in"', builtin, builtin ? await text(page, '.result') : '')
  check('ASL: taught signs never appear', (await page.locator('.badge.taught').count()) === 0)
  check('ASL: built-in score bars are back', (await page.locator('.score-row').count()) === 4)
  const sentence = await page.waitForFunction(() => /drink/.test(document.querySelector('.sentence p')?.textContent ?? ''), null, { timeout: 30000 }).then(() => true, () => false)
  check('ASL: built-in "drink" goes into the sentence', sentence, await text(page, '.sentence p'))
  check('no console errors', errors.length === 0, errors.join(' | '))
  await browser.close()
}

async function deleteSign(camera) {
  console.log('\nDelete a taught sign')
  const { browser, page } = await openApp(camera)
  await page.getByRole('button', { name: 'Teach', exact: true }).click()
  await page.locator('.taught-list li').first().waitFor({ timeout: 5000 })
  await page.locator('.taught-list li').getByRole('button', { name: 'Delete' }).first().click()
  await page.waitForFunction(() => document.querySelectorAll('.taught-list li').length === 0, null, { timeout: 5000 })
  check('Delete removes it from the list', true)
  check('and from the backend', (await (await fetch(`${API}/api/prototypes`)).json()).signs.length === 0)
  await browser.close()
}

// ------------------------------------------------------------------ main

const tmp = mkdtempSync(join(tmpdir(), 'signlang-teach-'))
const camera = join(tmp, 'camera.y4m')
try {
  execFileSync(PYTHON, ['-m', 'tests.fake_camera', 'data/raw/drink/17709.mp4', camera, '4'], { cwd: BACKEND })
  start(PYTHON, ['-m', 'uvicorn', 'app.main:app', '--port', '8000'], BACKEND,
    { TF_CPP_MIN_LOG_LEVEL: '2', FEWSHOT_PROTOTYPES: join(tmp, 'prototypes.json'), FEWSHOT_EVAL_CLIPS: join(tmp, 'eval') })
  start(process.execPath, [VITE, '--port', '5173', '--strictPort'], FRONTEND)
  await waitFor(`${API}/api/health`)
  await waitFor(APP)
  console.log('teach + modes ui test')

  await islEmptyState(camera)
  await teachFlow(camera)
  await modes(camera)
  await deleteSign(camera)
} catch (e) {
  failures.push(`crashed: ${e.message}`)
  console.error(e)
} finally {
  children.forEach((c) => c.kill())
  rmSync(tmp, { recursive: true, force: true })
}

console.log(failures.length ? `\n${failures.length} check(s) failed: ${failures.join(', ')}` : '\nall teach/mode ui checks passed')
process.exit(failures.length ? 1 : 0)
