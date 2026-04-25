declare module 'three/addons/renderers/webgpu/WebGPURenderer.js' {
  import type { Scene, Camera, ColorRepresentation, WebGLRenderTarget } from 'three'

  export default class WebGPURenderer {
    readonly isWebGPURenderer: boolean
    domElement: HTMLCanvasElement
    shadowMap: { enabled: boolean; type: number | null }
    outputColorSpace: string
    toneMapping: number
    toneMappingExposure: number

    constructor(params?: {
      canvas?: HTMLCanvasElement | OffscreenCanvas
      antialias?: boolean
      alpha?: boolean
      forceWebGL?: boolean
    })

    setClearColor(color: ColorRepresentation, alpha?: number): void
    setSize(width: number, height: number, updateStyle?: boolean): void
    setPixelRatio(value: number): void
    setRenderTarget(target: WebGLRenderTarget | null): void
    render(scene: Scene, camera: Camera): Promise<void>
    dispose(): void
    getContext(): WebGL2RenderingContext | GPUDevice
  }
}
