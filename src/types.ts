export type Vec3 = [number, number, number]

export type CameraCoordinates = {
  position: Vec3
  target: Vec3
  zoom: number
}

export type ObjectTransform = {
  position: Vec3
  scale: number
}

export type ScreenTransform = ObjectTransform & {
  aspectRatio: [number, number]
  borderRadius: number
  curvePair: 'horizontal' | 'vertical'
  edgeCurve: {
    top: number
    bottom: number
    left: number
    right: number
  }
  contentScale: number
}

export type SceneLayout = {
  orb: ObjectTransform
  screen: ScreenTransform
  groupRotation: number
  worldSize: number
}

export type HdrType = string | null
export type RendererType = 'webgl2' | 'webgpu'
export type OrbLighting = boolean
export type GroundGrid = 1 | 2 | 4 | 6 | 8