export interface SceneBridgeInstance {
  setSize: (w: number, h: number) => void
  setLayout: (layout: unknown) => void
  setCamera: (camera: unknown) => void
  setVideoUrl: (url: string) => void
  setEnergy: (energy: number) => void
  getPixels: () => string | null
  isReady: () => boolean
}

export function initSceneBridge(
  canvasEl: HTMLCanvasElement,
  _onLayoutChange: (layout: unknown) => void,
  _onCameraChange: (camera: unknown) => void
): SceneBridgeInstance {
  const bridgeInstance: SceneBridgeInstance = {
    setSize: (w: number, h: number) => {
      canvasEl.width = w
      canvasEl.height = h
      canvasEl.getContext('webgl2', { preserveDrawingBuffer: true })
      canvasEl.getContext('webgl', { preserveDrawingBuffer: true })
    },
    setLayout: (layout: unknown) => {
      window.dispatchEvent(new CustomEvent('__bridge_setLayout', { detail: layout }))
    },
    setCamera: (camera: unknown) => {
      window.dispatchEvent(new CustomEvent('__bridge_setCamera', { detail: camera }))
    },
    setVideoUrl: (url: string) => {
      window.dispatchEvent(new CustomEvent('__bridge_setVideoUrl', { detail: url }))
    },
    setEnergy: (energy: number) => {
      window.dispatchEvent(new CustomEvent('__bridge_setEnergy', { detail: energy }))
    },
    getPixels: () => {
      try {
        const w = canvasEl.width
        const h = canvasEl.height
        const gl = canvasEl.getContext('webgl2') || canvasEl.getContext('webgl')
        if (!gl) return null
        const pixels = new Uint8Array(w * h * 4)
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        const ctx = canvas.getContext('2d')
        if (!ctx) return null
        const imgData = new ImageData(new Uint8ClampedArray(pixels), w, h)
        ctx.putImageData(imgData, 0, h)
        ctx.scale(1, -1)
        ctx.drawImage(canvas, 0, 0)
        return canvas.toDataURL('image/jpeg', 0.8).split(',')[1] || null
      } catch (e) {
        console.error('getPixels error:', e)
        return null
      }
    },
    isReady: () => true,
  }

  ;(window as unknown as { __SCENE_BRIDGE__: SceneBridgeInstance }).__SCENE_BRIDGE__ = bridgeInstance

  return bridgeInstance
}