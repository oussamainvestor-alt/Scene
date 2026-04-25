import { useGLTF } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import {
  Group,
  LinearFilter,
  Matrix4,
  Mesh,
  PerspectiveCamera,
  Plane,
  RGBAFormat,
  ShaderMaterial,
  Vector3,
  WebGLRenderTarget,
} from 'three'
import type { GroundGrid } from '../../types'

const REFL_RES = 1024

const GROUND_VERT = `
  varying vec3 vWorldPos;
  varying vec4 vClipPos;
  varying vec2 vUv;

  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    vClipPos  = projectionMatrix * viewMatrix * wp;
    vUv = uv;
    gl_Position = vClipPos;
  }
`

const GROUND_FRAG = `
  precision highp float;
  #define PI 3.14159265359

  varying vec3 vWorldPos;
  varying vec4 vClipPos;
  varying vec2 vUv;

  uniform sampler2D tReflection;
  uniform sampler2D tBase;
  uniform float     uTime;
  uniform vec3      uCamPos;

  const vec3 DIR_L     = normalize(vec3(2.5, 4.2, 1.8));
  const vec3 DIR_COL   = vec3(0.478, 0.596, 0.753);
  const vec3 SPOT_POS  = vec3(0.0, 5.8, 0.6);
  const vec3 SPOT_COL  = vec3(0.710, 0.788, 0.937);
  const vec3 POINT_POS = vec3(-2.5, 1.4, -2.0);
  const vec3 POINT_COL = vec3(0.255, 0.380, 0.561);

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i), hash(i+vec2(1,0)), f.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
  }
  float fbm(vec2 p, int oct) {
    float v=0.0, a=0.5, fr=1.0;
    for(int i=0;i<8;i++){if(i>=oct)break; v+=a*vnoise(p*fr); a*=0.5; fr*=2.0;}
    return v;
  }

  float D_GGX(float nh, float r) { float a2=r*r*r*r; float d=nh*nh*(a2-1.0)+1.0; return a2/(PI*d*d+1e-6); }
  float G_Smith(float nv, float nl, float r) { float k=(r+1.0)*(r+1.0)/8.0; return (nv/(nv*(1.0-k)+k))*(nl/(nl*(1.0-k)+k)); }
  vec3 F_Schlick(float c, vec3 F0) { return F0+(1.0-F0)*pow(clamp(1.0-c,0.0,1.0),5.0); }
  vec3 cookTorrance(vec3 N, vec3 V, vec3 L, vec3 lc, vec3 alb, float r, vec3 F0) {
    vec3 H=normalize(V+L);
    float nl=max(dot(N,L),0.0), nv=max(dot(N,V),0.001), nh=max(dot(N,H),0.0), hv=max(dot(H,V),0.0);
    vec3 F=F_Schlick(hv,F0);
    vec3 sp=D_GGX(nh,r)*G_Smith(nv,nl,r)*F/(4.0*nv*nl+1e-6);
    return ((1.0-F)*alb/PI+sp)*lc*nl;
  }
  vec3 surfaceNormal(vec2 uv, float str) {
    float e=0.007, h0=fbm(uv,4), h1=fbm(uv+vec2(e,0),4), h2=fbm(uv+vec2(0,e),4);
    return normalize(vec3(-(h1-h0)/e*str, 1.0, -(h2-h0)/e*str));
  }

  void main() {
    vec2 worldUv = vWorldPos.xz / 5.0;

    float pLarge  = fbm(worldUv*0.38, 4);
    float pDetail = fbm(worldUv*1.15+vec2(3.7,1.3), 3);
    float puddle  = smoothstep(0.43, 0.57, pLarge*0.65+pDetail*0.35);

    // Base concrete texture from GLTF (sRGB -> linear)
    vec3 baseColor = texture2D(tBase, vUv).rgb;
    baseColor = pow(max(baseColor, vec3(0.001)), vec3(2.2));

    // Animated ripple normals inside puddles
    vec2 rA = worldUv*2.6+vec2(uTime*0.024,  uTime*0.017);
    vec2 rB = worldUv*2.6+vec2(-uTime*0.013, -uTime*0.029);
    float e=0.007;
    float r0=fbm(rA,3)*0.7+fbm(rB,3)*0.3;
    float r1=fbm(rA+vec2(e,0),3)*0.7+fbm(rB+vec2(e,0),3)*0.3;
    float r2=fbm(rA+vec2(0,e),3)*0.7+fbm(rB+vec2(0,e),3)*0.3;
    vec3 rippleN = normalize(vec3(-(r1-r0)/e*0.10, 1.0, -(r2-r0)/e*0.10));
    vec3 asphN   = surfaceNormal(worldUv*4.5, 0.38);
    vec3 N = normalize(mix(asphN, rippleN, puddle));

    // Wet darkens concrete; puddles are mirror-smooth
    vec3 albedo = mix(baseColor, baseColor*0.32, puddle);
    float roughness = mix(0.84, 0.022, puddle);
    vec3  F0 = mix(vec3(0.04), vec3(0.020, 0.021, 0.026), puddle);

    vec3 V = normalize(uCamPos - vWorldPos);

    // Screen-space planar reflection (perturbed by ripple normal)
    vec2 scrUv   = (vClipPos.xy/vClipPos.w)*0.5+0.5;
    vec2 distort = N.xz*puddle*0.020;
    vec2 reflUv  = clamp(scrUv+distort, vec2(0.001), vec2(0.999));
    vec3 reflCol = texture2D(tReflection, reflUv).rgb;
    float fresnel = F_Schlick(max(dot(N,V),0.0), F0).g;
    float reflStr = clamp(mix(0.0, fresnel*2.4, puddle), 0.0, 0.96);

    // Direct lighting (Cook-Torrance GGX)
    vec3 Lo = vec3(0.0);
    Lo += cookTorrance(N,V,DIR_L,DIR_COL*0.55,albedo,roughness,F0);
    vec3 sDir=normalize(SPOT_POS-vWorldPos); float sDst=length(SPOT_POS-vWorldPos);
    Lo += cookTorrance(N,V,sDir,SPOT_COL/(1.0+0.09*sDst+0.018*sDst*sDst)*2.4,albedo,roughness,F0);
    vec3 pDir=normalize(POINT_POS-vWorldPos); float pDst=length(POINT_POS-vWorldPos);
    Lo += cookTorrance(N,V,pDir,POINT_COL/(1.0+0.22*pDst+0.07*pDst*pDst)*1.5,albedo,roughness,F0);

    vec3 color = albedo*vec3(0.05,0.065,0.10) + Lo;
    color = mix(color, reflCol, reflStr);

    // Fog (matches scene: near=5, far=18)
    float fogDist = length(vWorldPos-uCamPos);
    float fog = clamp((fogDist-5.0)/13.0, 0.0, 1.0);
    color = mix(color, vec3(0.024,0.039,0.067), fog);

    gl_FragColor = vec4(color, 1.0);
  }
`

function getTilePositions(groundGrid: GroundGrid): Array<[number, number, number]> {
  if (groundGrid === 1) return [[0, 0, 0]]
  if (groundGrid === 2) return [[-5, 0, -5], [5, 0, -5], [-5, 0, 5], [5, 0, 5]]
  const n = groundGrid
  const half = (n - 1) / 2
  const positions: Array<[number, number, number]> = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      positions.push([(i - half) * 5, 0, (j - half) * 5])
  return positions
}

type GroundTileProps = {
  position: [number, number, number]
  material: ShaderMaterial
}

function GroundTile({ position, material }: GroundTileProps) {
  const { scene } = useGLTF('/ground_concrete.glb')
  const cloned = useMemo(() => scene.clone(), [scene])

  useEffect(() => {
    cloned.traverse((child) => {
      if (child instanceof Mesh) {
        child.material = material
        child.receiveShadow = true
      }
    })
  }, [cloned, material])

  return <primitive object={cloned} position={position} />
}

type WetAsphaltGroundProps = {
  groundGrid?: GroundGrid
}

export function WetAsphaltGround({ groundGrid = 1 }: WetAsphaltGroundProps) {
  const { scene: threeScene } = useThree()
  const groupRef = useRef<Group>(null)

  const reflTarget = useMemo(() => new WebGLRenderTarget(REFL_RES, REFL_RES, {
    minFilter: LinearFilter,
    magFilter: LinearFilter,
    format: RGBAFormat,
  }), [])

  const virtualCam = useMemo(() => {
    const cam = new PerspectiveCamera()
    cam.matrixAutoUpdate = false
    return cam
  }, [])

  const reflMatrix      = useMemo(() => new Matrix4().makeScale(1, -1, 1), [])
  const groundClipPlane = useMemo(() => new Plane(new Vector3(0, 1, 0), 0), [])

  const { scene: gltfScene } = useGLTF('/ground_concrete.glb')

  const baseTexture = useMemo((): unknown => {
    let tex: unknown = null
    gltfScene.traverse((child) => {
      if (child instanceof Mesh && (child.material as any)?.map && !tex) {
        tex = (child.material as any).map
      }
    })
    return tex
  }, [gltfScene])

  const material = useMemo(() => new ShaderMaterial({
    uniforms: {
      tReflection: { value: reflTarget.texture },
      tBase:       { value: baseTexture },
      uTime:       { value: 0 },
      uCamPos:     { value: new Vector3() },
    },
    vertexShader:   GROUND_VERT,
    fragmentShader: GROUND_FRAG,
    toneMapped: false,
  }), [reflTarget, baseTexture])

  const tiles = useMemo(() => getTilePositions(groundGrid), [groundGrid])

  useFrame(({ gl, camera, clock }) => {
    if (!groupRef.current) return

    virtualCam.projectionMatrix.copy(camera.projectionMatrix)
    virtualCam.projectionMatrixInverse.copy(camera.projectionMatrixInverse)
    virtualCam.matrixWorld.copy(camera.matrixWorld).premultiply(reflMatrix)
    virtualCam.matrixWorldInverse.copy(virtualCam.matrixWorld).invert()

    groupRef.current.visible = false
    const prevTarget     = gl.getRenderTarget()
    const prevLocalClip  = gl.localClippingEnabled
    const prevClipPlanes = gl.clippingPlanes

    gl.localClippingEnabled = true
    gl.clippingPlanes = [groundClipPlane]
    gl.setRenderTarget(reflTarget)
    gl.clear()
    gl.render(threeScene, virtualCam)
    gl.setRenderTarget(prevTarget)
    gl.clippingPlanes = prevClipPlanes
    gl.localClippingEnabled = prevLocalClip
    groupRef.current.visible = true

    material.uniforms.uTime.value = clock.elapsedTime
    material.uniforms.uCamPos.value.copy(camera.position)
  })

  return (
    <group ref={groupRef}>
      {tiles.map((pos, i) => (
        <GroundTile key={i} position={pos} material={material} />
      ))}
    </group>
  )
}
