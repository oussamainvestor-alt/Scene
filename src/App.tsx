import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { ControlPanel } from './components/ControlPanel'
import { LayoutPanel } from './components/LayoutPanel'
import { EnvironmentScene } from './components/scene/EnvironmentScene'
import type { EnvironmentSceneHandle } from './components/scene/EnvironmentScene'
import { useAudioDriver } from './hooks/useAudioDriver'
import { useRecorder } from './hooks/useRecorder'
import type { CameraCoordinates, SceneLayout, HdrType, RendererType, OrbLighting, GroundGrid } from './types'

export const HDR_FILES: Array<{ label: string; value: string }> = [
  { label: 'wooden_studio_04_4k.hdr', value: 'wooden_studio_04_4k.hdr' },
  { label: 'church_stairway_4k.hdr', value: 'church_stairway_4k.hdr' },
  { label: 'university_workshop_4k.hdr', value: 'university_workshop_4k.hdr' },
]

const DEFAULT_LAYOUT: SceneLayout = {
  orb: { position: [0, 1.35, 0], scale: 1 },
  screen: {
    position: [0, 1.65, -2.4],
    scale: 1,
    aspectRatio: [16, 9],
    borderRadius: 0.08,
    curvePair: 'horizontal',
    edgeCurve: {
      top: -0.08,
      bottom: -0.08,
      left: 0,
      right: 0,
    },
  },
  objectReflection: 0.62,
  objectReflectionOpacity: 0.9,
  groundSurface: 0.5,
  groupRotation: 0,
}

function normalizeLayout(layout: Partial<SceneLayout>): SceneLayout {
  const legacy = layout as Partial<SceneLayout> & { groundReflection?: number }
  const next = structuredClone(DEFAULT_LAYOUT)

  if (layout.orb) {
    next.orb = {
      position: layout.orb.position ?? next.orb.position,
      scale: layout.orb.scale ?? next.orb.scale,
    }
  }

  if (layout.screen) {
    next.screen = {
      position: layout.screen.position ?? next.screen.position,
      scale: layout.screen.scale ?? next.screen.scale,
      aspectRatio: layout.screen.aspectRatio ?? next.screen.aspectRatio,
      borderRadius: layout.screen.borderRadius ?? next.screen.borderRadius,
      curvePair: layout.screen.curvePair ?? next.screen.curvePair,
      edgeCurve: {
        top: layout.screen.edgeCurve?.top ?? next.screen.edgeCurve.top,
        bottom: layout.screen.edgeCurve?.bottom ?? next.screen.edgeCurve.bottom,
        left: layout.screen.edgeCurve?.left ?? next.screen.edgeCurve.left,
        right: layout.screen.edgeCurve?.right ?? next.screen.edgeCurve.right,
      },
    }
  }

  next.objectReflection =
    layout.objectReflection ?? legacy.groundReflection ?? next.objectReflection
  next.objectReflectionOpacity = layout.objectReflectionOpacity ?? next.objectReflectionOpacity
  next.groundSurface = layout.groundSurface ?? legacy.groundReflection ?? next.groundSurface
  next.groupRotation = layout.groupRotation ?? next.groupRotation

  return next
}

type EditorSnapshot = {
  layout: SceneLayout
  camera: CameraCoordinates
  activeCameraPreset: CameraPresetId
  dragTarget: 'none' | 'orb' | 'screen' | 'ground'
}

const DEFAULT_CAMERA: CameraCoordinates = {
  position: [4.2, 2.2, 4.4],
  target: [0, 1.2, 0],
  zoom: 1,
}

function snapshotsEqual(a: EditorSnapshot, b: EditorSnapshot) {
  return JSON.stringify(a) === JSON.stringify(b)
}

function cloneSnapshot(snapshot: EditorSnapshot): EditorSnapshot {
  return {
    layout: structuredClone(snapshot.layout),
    camera: structuredClone(snapshot.camera),
    activeCameraPreset: snapshot.activeCameraPreset,
    dragTarget: snapshot.dragTarget,
  }
}

function downloadJsonFile(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

type CameraPresetId = 'frontCenter' | 'orbFocus' | 'rightAngle' | 'leftAngle' | 'topView'

function getCameraPresetCoordinates(preset: CameraPresetId, layout: SceneLayout): CameraCoordinates {
  const orb = layout.orb.position
  const screen = layout.screen.position
  const zGap = Math.abs(orb[2] - screen[2])

  switch (preset) {
    case 'frontCenter':
      // Front-facing, dead-center to screen, camera behind orb.
      return {
        position: [screen[0], screen[1], orb[2] + Math.max(2.7, zGap + 1.1)],
        target: [screen[0], screen[1], screen[2]],
        zoom: 1,
      }
    case 'orbFocus':
      return {
        position: [orb[0], orb[1] + 0.45, orb[2] + 2.2],
        target: [orb[0], orb[1], orb[2]],
        zoom: 1.05,
      }
    case 'rightAngle':
      return {
        position: [screen[0] + 2.2, screen[1] + 0.22, orb[2] + 2.0],
        target: [screen[0], screen[1], screen[2]],
        zoom: 1,
      }
    case 'leftAngle':
      return {
        position: [screen[0] - 2.2, screen[1] + 0.22, orb[2] + 2.0],
        target: [screen[0], screen[1], screen[2]],
        zoom: 1,
      }
    case 'topView':
      return {
        position: [screen[0], screen[1] + 3.4, orb[2] + 0.8],
        target: [screen[0], screen[1] - 0.25, screen[2] + 0.5],
        zoom: 1,
      }
  }
}

function App() {
  const sceneRef = useRef<EnvironmentSceneHandle | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [videoFileName, setVideoFileName] = useState('No video selected')
  const [audioFileName, setAudioFileName] = useState('No audio selected')
  const [dragTarget, setDragTarget] = useState<'none' | 'orb' | 'screen' | 'ground'>('none')
  const [layout, setLayout] = useState<SceneLayout>(DEFAULT_LAYOUT)
  const [activeCameraPreset, setActiveCameraPreset] = useState<CameraPresetId>('frontCenter')
  const [cameraCoordinates, setCameraCoordinates] = useState<CameraCoordinates>(DEFAULT_CAMERA)
  const [rendererType, setRendererType] = useState<RendererType>('webgpu')
  const [hdrType, setHdrType] = useState<HdrType>(null)
  const [orbLighting, setOrbLighting] = useState<OrbLighting>(true)
  const [groundGrid, setGroundGrid] = useState<GroundGrid>(1)
  const currentSnapshotRef = useRef<EditorSnapshot>({
    layout: structuredClone(DEFAULT_LAYOUT),
    camera: structuredClone(DEFAULT_CAMERA),
    activeCameraPreset: 'frontCenter',
    dragTarget: 'none',
  })
  const undoStackRef = useRef<EditorSnapshot[]>([])
  const redoStackRef = useRef<EditorSnapshot[]>([])

  const { sourceLabel, level, useMicrophone, selectAudioFile, setMicMode } = useAudioDriver()
  const {
    isRecording,
    recordedUrl,
    recordedFileName,
    setCanvas,
    startRecording,
    stopRecording,
    clearRecording,
  } = useRecorder()

  const orbEnergy = useMemo(() => Math.min(1, level * 2.2), [level])

  const applySnapshot = (snapshot: EditorSnapshot) => {
    const cloned = cloneSnapshot(snapshot)
    currentSnapshotRef.current = cloned
    setLayout(cloned.layout)
    setCameraCoordinates(cloned.camera)
    setActiveCameraPreset(cloned.activeCameraPreset)
    setDragTarget(cloned.dragTarget)
  }

  const commitSnapshot = (next: EditorSnapshot) => {
    const current = currentSnapshotRef.current
    if (snapshotsEqual(current, next)) {
      return
    }
    undoStackRef.current.push(cloneSnapshot(current))
    if (undoStackRef.current.length > 200) {
      undoStackRef.current = undoStackRef.current.slice(-200)
    }
    redoStackRef.current = []
    applySnapshot(next)
  }

  useEffect(() => {
    const handleUndoRedo = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) {
        return
      }
      const target = event.target
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT')
      ) {
        return
      }
      const key = event.key.toLowerCase()
      if (key === 'z') {
        event.preventDefault()
        const previous = undoStackRef.current.pop()
        if (!previous) return
        redoStackRef.current.push(cloneSnapshot(currentSnapshotRef.current))
        if (redoStackRef.current.length > 200) {
          redoStackRef.current = redoStackRef.current.slice(-200)
        }
        applySnapshot(previous)
        return
      }
      if (key === 'y') {
        event.preventDefault()
        const next = redoStackRef.current.pop()
        if (!next) return
        undoStackRef.current.push(cloneSnapshot(currentSnapshotRef.current))
        if (undoStackRef.current.length > 200) {
          undoStackRef.current = undoStackRef.current.slice(-200)
        }
        applySnapshot(next)
      }
    }

    window.addEventListener('keydown', handleUndoRedo)
    return () => window.removeEventListener('keydown', handleUndoRedo)
  }, [])

  const handleVideoUpload = (file: File | null) => {
    if (!file) {
      return
    }

    setVideoFileName(file.name)
    const nextUrl = URL.createObjectURL(file)
    setVideoUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous)
      }
      return nextUrl
    })
  }

  const handleAudioUpload = async (file: File | null) => {
    if (!file) {
      return
    }
    setAudioFileName(file.name)
    await selectAudioFile(file)
  }

  const handleCoordinatesUpload = async (file: File | null) => {
    if (!file || !sceneRef.current) {
      return
    }
    const raw = await file.text()
    const parsed = JSON.parse(raw) as CameraCoordinates
    commitSnapshot({
      ...cloneSnapshot(currentSnapshotRef.current),
      camera: parsed,
      activeCameraPreset: 'frontCenter',
    })
  }

  const handleCoordinatesDownload = () => {
    if (!sceneRef.current) {
      return
    }
    const coords = sceneRef.current.getCoordinates()
    downloadJsonFile(coords, 'camera-coordinates.json')
  }

  const handleLayoutDownload = () => {
    downloadJsonFile(layout, 'scene-layout.json')
  }

  const handleLayoutUpload = async (file: File | null) => {
    if (!file) return
    const raw = await file.text()
    const parsed = JSON.parse(raw) as Partial<SceneLayout>
    commitSnapshot({
      ...cloneSnapshot(currentSnapshotRef.current),
      layout: normalizeLayout(parsed),
    })
  }

  const applyCameraPreset = (preset: CameraPresetId) => {
    const next = getCameraPresetCoordinates(preset, currentSnapshotRef.current.layout)
    commitSnapshot({
      ...cloneSnapshot(currentSnapshotRef.current),
      camera: next,
      activeCameraPreset: preset,
    })
  }

  const handleLayoutChange = (next: SceneLayout) => {
    commitSnapshot({
      ...cloneSnapshot(currentSnapshotRef.current),
      layout: next,
    })
  }

  const handleDragTargetChange = (next: 'none' | 'orb' | 'screen' | 'ground') => {
    commitSnapshot({
      ...cloneSnapshot(currentSnapshotRef.current),
      dragTarget: next,
    })
  }

  const handleCameraCoordinatesChange = (next: CameraCoordinates) => {
    commitSnapshot({
      ...cloneSnapshot(currentSnapshotRef.current),
      camera: next,
      activeCameraPreset: 'frontCenter',
    })
  }

  return (
    <div className="app-shell">
      <ControlPanel
        isRecording={isRecording}
        useMicrophone={useMicrophone}
        sourceLabel={sourceLabel}
        videoFileName={videoFileName}
        audioFileName={audioFileName}
        recordedUrl={recordedUrl}
        recordedFileName={recordedFileName}
        onToggleMic={setMicMode}
        onVideoUpload={handleVideoUpload}
        onAudioUpload={handleAudioUpload}
        onCoordinateUpload={handleCoordinatesUpload}
        onCoordinatesDownload={handleCoordinatesDownload}
        onStartRecording={startRecording}
        onStopRecording={stopRecording}
        onClearRecording={clearRecording}
      />
      <LayoutPanel
        layout={layout}
        dragTarget={dragTarget}
        onDragTargetChange={handleDragTargetChange}
        onLayoutChange={handleLayoutChange}
        rendererType={rendererType}
        onRendererTypeChange={setRendererType}
        hdrType={hdrType}
        onHdrTypeChange={setHdrType}
        hdrFiles={HDR_FILES}
        orbLighting={orbLighting}
        onOrbLightingChange={setOrbLighting}
        groundGrid={groundGrid}
        onGroundGridChange={setGroundGrid}
        groupRotation={layout.groupRotation}
        onGroupRotationChange={(r) => handleLayoutChange({ ...layout, groupRotation: r })}
        onDownloadLayout={handleLayoutDownload}
        onUploadLayout={handleLayoutUpload}
      />
      <aside
        className="camera-presets"
        style={{
          position: 'fixed',
          left: '12px',
          bottom: '12px',
          width: '220px',
          zIndex: 40,
        }}
      >
        <p className="camera-presets-title">Default Camera Positions</p>
        <div className="camera-presets-grid">
          <button
            className={`btn secondary ${activeCameraPreset === 'frontCenter' ? 'is-active' : ''}`}
            onClick={() => applyCameraPreset('frontCenter')}
          >
            Front Center
          </button>
          <button
            className={`btn secondary ${activeCameraPreset === 'orbFocus' ? 'is-active' : ''}`}
            onClick={() => applyCameraPreset('orbFocus')}
          >
            Orb Focus
          </button>
          <button
            className={`btn secondary ${activeCameraPreset === 'rightAngle' ? 'is-active' : ''}`}
            onClick={() => applyCameraPreset('rightAngle')}
          >
            Right Angle
          </button>
          <button
            className={`btn secondary ${activeCameraPreset === 'leftAngle' ? 'is-active' : ''}`}
            onClick={() => applyCameraPreset('leftAngle')}
          >
            Left Angle
          </button>
          <button
            className={`btn secondary ${activeCameraPreset === 'topView' ? 'is-active' : ''}`}
            onClick={() => applyCameraPreset('topView')}
          >
            Top View
          </button>
        </div>
      </aside>

      <EnvironmentScene
        ref={sceneRef}
        videoUrl={videoUrl}
        orbEnergy={orbEnergy}
        orbLighting={orbLighting}
        rendererType={rendererType}
        hdrType={hdrType}
        groundGrid={groundGrid}
        onCanvasReady={setCanvas}
        layout={layout}
        dragTarget={dragTarget}
        onLayoutChange={handleLayoutChange}
        cameraCoordinates={cameraCoordinates}
        onCameraCoordinatesChange={handleCameraCoordinatesChange}
      />
    </div>
  )
}

export default App
