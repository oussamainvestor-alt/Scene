import { useState, type MouseEvent as ReactMouseEvent } from 'react'
import { createPortal } from 'react-dom'
import type { SceneLayout, HdrType, RendererType, OrbLighting, GroundGrid } from '../types'

type LayoutPanelProps = {
  layout: SceneLayout
  dragTarget: 'none' | 'orb' | 'screen' | 'ground'
  onDragTargetChange: (next: 'none' | 'orb' | 'screen' | 'ground') => void
  onLayoutChange: (next: SceneLayout) => void
  rendererType?: RendererType
  onRendererTypeChange?: (next: RendererType) => void
  hdrType?: HdrType
  onHdrTypeChange?: (next: HdrType) => void
  hdrFiles?: Array<{ label: string; value: string }>
  orbLighting?: OrbLighting
  onOrbLightingChange?: (next: OrbLighting) => void
  groundGrid?: GroundGrid
  onGroundGridChange?: (next: GroundGrid) => void
  groupRotation?: number
  onGroupRotationChange?: (next: number) => void
  onDownloadLayout: () => void
  onUploadLayout?: (file: File | null) => void
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function LayoutPanel(props: LayoutPanelProps) {
  const [offset, setOffset] = useState({ x: 0, y: 0 })

  const beginDrag = (event: ReactMouseEvent<HTMLDivElement>) => {
    const startX = event.clientX
    const startY = event.clientY
    const initial = { ...offset }

    const onMove = (moveEvent: MouseEvent) => {
      setOffset({
        x: initial.x + (moveEvent.clientX - startX),
        y: initial.y + (moveEvent.clientY - startY),
      })
    }

    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const setPos = (
    objectKey: 'orb' | 'screen',
    axis: 0 | 1 | 2,
    value: number,
  ) => {
    const next = structuredClone(props.layout)
    next[objectKey].position[axis] = value
    props.onLayoutChange(next)
  }

  const setScale = (objectKey: 'orb' | 'screen', value: number) => {
    const next = structuredClone(props.layout)
    next[objectKey].scale = clamp(value, 0.2, 4)
    props.onLayoutChange(next)
  }

  const setScreenAspectRatio = (index: 0 | 1, value: number) => {
    const next = structuredClone(props.layout)
    const safeValue = clamp(value || 1, 1, 1000)
    next.screen.aspectRatio[index] = safeValue
    props.onLayoutChange(next)
  }

  const setScreenBorderRadius = (value: number) => {
    const next = structuredClone(props.layout)
    next.screen.borderRadius = clamp(value || 0, 0, 1)
    props.onLayoutChange(next)
  }

  const setScreenCurvePair = (value: 'horizontal' | 'vertical') => {
    const next = structuredClone(props.layout)
    next.screen.curvePair = value
    props.onLayoutChange(next)
  }

  const setScreenEdgeCurve = (side: 'top' | 'bottom' | 'left' | 'right', value: number) => {
    const next = structuredClone(props.layout)
    next.screen.edgeCurve[side] = clamp(value || 0, -0.6, 0.6)
    props.onLayoutChange(next)
  }

  const setObjectReflection = (value: number) => {
    const next = structuredClone(props.layout)
    next.objectReflection = clamp(value || 0, 0, 1)
    props.onLayoutChange(next)
  }

  const setObjectReflectionOpacity = (value: number) => {
    const next = structuredClone(props.layout)
    next.objectReflectionOpacity = clamp(value || 0, 0, 1)
    props.onLayoutChange(next)
  }

  const setGroundSurface = (value: number) => {
    const next = structuredClone(props.layout)
    next.groundSurface = clamp(value || 0, 0, 1)
    props.onLayoutChange(next)
  }

  const curvePair = props.layout.screen.curvePair
  const firstCurveKey = curvePair === 'horizontal' ? 'top' : 'left'
  const secondCurveKey = curvePair === 'horizontal' ? 'bottom' : 'right'

  const panel = (
    <aside
      className="layout-panel"
      style={{
        position: 'fixed',
        top: '16px',
        right: '16px',
        zIndex: 9999,
        width: '260px',
        maxWidth: '280px',
        maxHeight: 'calc(100vh - 20px)',
        overflow: 'auto',
        borderRadius: '12px',
        border: '1px solid rgba(255, 255, 255, 0.15)',
        background: 'rgba(10, 14, 24, 0.84)',
        backdropFilter: 'blur(12px)',
        color: '#eef3ff',
        padding: '10px',
        boxShadow: '0 8px 22px rgba(0, 0, 0, 0.42)',
        boxSizing: 'border-box',
        transform: `translate3d(${offset.x}px, ${offset.y}px, 0)`,
      }}
    >
      <div className="panel-drag-handle" onMouseDown={beginDrag}>
        <h3>Layout Controls</h3>
      </div>
      <div className="layout-row">
        <button className="btn secondary" onClick={() => props.onDragTargetChange('ground')}>
          Drag ground
        </button>
        <button className="btn secondary" onClick={() => props.onDragTargetChange('orb')}>
          Drag orb
        </button>
        <button className="btn secondary" onClick={() => props.onDragTargetChange('screen')}>
          Drag screen
        </button>
      </div>
      <button className="btn secondary full" onClick={() => props.onDragTargetChange('none')}>
        Disable drag ({props.dragTarget})
      </button>

      <div className="divider" />
      <p className="tiny-title">Orb</p>
      <div className="compact-grid">
        <input
          type="number"
          step="0.1"
          value={props.layout.orb.position[0]}
          onChange={(e) => setPos('orb', 0, Number(e.target.value))}
        />
        <input
          type="number"
          step="0.1"
          value={props.layout.orb.position[1]}
          onChange={(e) => setPos('orb', 1, Number(e.target.value))}
        />
        <input
          type="number"
          step="0.1"
          value={props.layout.orb.position[2]}
          onChange={(e) => setPos('orb', 2, Number(e.target.value))}
        />
      </div>
      <div className="scale-row">
        <input
          type="range"
          min="0.4"
          max="2.4"
          step="0.01"
          value={props.layout.orb.scale}
          onChange={(e) => setScale('orb', Number(e.target.value))}
        />
        <input
          type="number"
          min="0.4"
          max="2.4"
          step="0.01"
          value={props.layout.orb.scale}
          onChange={(e) => setScale('orb', Number(e.target.value))}
        />
      </div>

      <p className="tiny-title">Screen</p>
      <div className="compact-grid">
        <input
          type="number"
          step="0.1"
          value={props.layout.screen.position[0]}
          onChange={(e) => setPos('screen', 0, Number(e.target.value))}
        />
        <input
          type="number"
          step="0.1"
          value={props.layout.screen.position[1]}
          onChange={(e) => setPos('screen', 1, Number(e.target.value))}
        />
        <input
          type="number"
          step="0.1"
          value={props.layout.screen.position[2]}
          onChange={(e) => setPos('screen', 2, Number(e.target.value))}
        />
      </div>
      <div className="scale-row">
        <input
          type="range"
          min="0.4"
          max="2.4"
          step="0.01"
          value={props.layout.screen.scale}
          onChange={(e) => setScale('screen', Number(e.target.value))}
        />
        <input
          type="number"
          min="0.4"
          max="2.4"
          step="0.01"
          value={props.layout.screen.scale}
          onChange={(e) => setScale('screen', Number(e.target.value))}
        />
      </div>
      <p className="tiny-title">Aspect Ratio (W : H)</p>
      <div className="scale-row">
        <input
          type="number"
          min="1"
          step="1"
          value={props.layout.screen.aspectRatio[0]}
          onChange={(e) => setScreenAspectRatio(0, Number(e.target.value))}
        />
        <input
          type="number"
          min="1"
          step="1"
          value={props.layout.screen.aspectRatio[1]}
          onChange={(e) => setScreenAspectRatio(1, Number(e.target.value))}
        />
      </div>
      <p className="tiny-title">Rounded Corners (Border Radius)</p>
      <div className="scale-row">
        <input
          type="range"
          min="0"
          max="0.5"
          step="0.01"
          value={props.layout.screen.borderRadius}
          onChange={(e) => setScreenBorderRadius(Number(e.target.value))}
        />
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.screen.borderRadius}
          onChange={(e) => setScreenBorderRadius(Number(e.target.value))}
        />
      </div>
      <p className="tiny-title">Curve Pair</p>
      <div className="layout-row">
        <button
          className="btn secondary"
          onClick={() => setScreenCurvePair('horizontal')}
          style={{ opacity: curvePair === 'horizontal' ? 1 : 0.65 }}
        >
          Top + Bottom
        </button>
        <button
          className="btn secondary"
          onClick={() => setScreenCurvePair('vertical')}
          style={{ opacity: curvePair === 'vertical' ? 1 : 0.65 }}
        >
          Left + Right
        </button>
      </div>
      <p className="tiny-title">
        {curvePair === 'horizontal' ? 'Top / Bottom Curve' : 'Left / Right Curve'} (inner/outer)
      </p>
      <div className="scale-row">
        <input
          type="number"
          min="-0.6"
          max="0.6"
          step="0.01"
          value={props.layout.screen.edgeCurve[firstCurveKey]}
          onChange={(e) => setScreenEdgeCurve(firstCurveKey, Number(e.target.value))}
        />
        <input
          type="number"
          min="-0.6"
          max="0.6"
          step="0.01"
          value={props.layout.screen.edgeCurve[secondCurveKey]}
          onChange={(e) => setScreenEdgeCurve(secondCurveKey, Number(e.target.value))}
        />
      </div>
      <p className="tiny-title">Reflection: Objects (Orb + Screen)</p>
      <div className="scale-row">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.objectReflection}
          onChange={(e) => setObjectReflection(Number(e.target.value))}
        />
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.objectReflection}
          onChange={(e) => setObjectReflection(Number(e.target.value))}
        />
      </div>
      <p className="tiny-title">Reflection: Objects Opacity</p>
      <div className="scale-row">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.objectReflectionOpacity}
          onChange={(e) => setObjectReflectionOpacity(Number(e.target.value))}
        />
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.objectReflectionOpacity}
          onChange={(e) => setObjectReflectionOpacity(Number(e.target.value))}
        />
      </div>
      <p className="tiny-title">Ground Surface Strength</p>
      <div className="scale-row">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.groundSurface}
          onChange={(e) => setGroundSurface(Number(e.target.value))}
        />
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={props.layout.groundSurface}
          onChange={(e) => setGroundSurface(Number(e.target.value))}
        />
      </div>

      {props.rendererType !== undefined && props.onRendererTypeChange && (
        <>
          <p className="tiny-title">Renderer</p>
          <select
            className="select-input"
            value={props.rendererType}
            onChange={(e) => props.onRendererTypeChange?.(e.target.value as RendererType)}
          >
            <option value="webgpu">WebGPU</option>
            <option value="webgl2">WebGL2</option>
          </select>
        </>
      )}

      {props.hdrType !== undefined && props.onHdrTypeChange && (
        <>
          <p className="tiny-title">HDR</p>
          <select
            className="select-input"
            value={props.hdrType ?? ''}
            onChange={(e) => props.onHdrTypeChange?.(e.target.value || null)}
          >
            <option value="">Off</option>
            {props.hdrFiles?.map((hdr) => (
              <option key={hdr.value} value={hdr.value}>
                {hdr.label}
              </option>
            ))}
          </select>
        </>
      )}

      {props.groundGrid !== undefined && props.onGroundGridChange && (
        <>
          <p className="tiny-title">Ground Tiles</p>
          <select
            className="select-input"
            value={props.groundGrid}
            onChange={(e) => props.onGroundGridChange?.(Number(e.target.value) as GroundGrid)}
          >
            <option value={1}>1×1</option>
            <option value={2}>2×2</option>
            <option value={4}>4×4</option>
            <option value={6}>6×6</option>
            <option value={8}>8×8</option>
          </select>
        </>
      )}

      {props.groupRotation !== undefined && props.onGroupRotationChange && (
        <>
          <p className="tiny-title">Rotate (degrees)</p>
          <input
            type="number"
            className="select-input"
            value={(props.groupRotation * 180 / Math.PI) % 360}
            onChange={(e) => props.onGroupRotationChange?.(Number(e.target.value) * Math.PI / 180)}
          />
        </>
      )}

      {props.orbLighting !== undefined && props.onOrbLightingChange && (
        <>
          <p className="tiny-title">Orb Light</p>
          <select
            className="select-input"
            value={props.orbLighting ? 'on' : 'off'}
            onChange={(e) => props.onOrbLightingChange?.(e.target.value === 'on')}
          >
            <option value="on">On</option>
            <option value="off">Off</option>
          </select>
        </>
      )}

      <div className="divider" />
      <div className="layout-row">
        <button className="btn secondary" onClick={props.onDownloadLayout}>
          Download
        </button>
        <label className="btn secondary upload-btn">
          Upload
          <input
            type="file"
            accept="application/json"
            onChange={(event) => {
              props.onUploadLayout?.(event.target.files?.[0] ?? null)
            }}
          />
        </label>
      </div>
    </aside>
  )

  return createPortal(panel, document.body)
}
