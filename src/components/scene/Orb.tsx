import { useFrame } from '@react-three/fiber'
import { forwardRef, useEffect, useRef } from 'react'
import {
  AdditiveBlending,
  DoubleSide,
  FrontSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  PointLight,
  ShaderMaterial,
  Vector3,
} from 'three'

import type { ObjectTransform } from '../../types'

type OrbProps = {
  energy: number
  transform: ObjectTransform
  lightEnabled?: boolean
}

/** Atom / neon reference: thin bright rings + hot core + haze + floor glow. */
const RING_COUNT = 6
const ringMajor = [0.11, 0.145, 0.125, 0.165, 0.098, 0.138]
const ringTube = [0.0045, 0.0038, 0.0042, 0.0035, 0.005, 0.004]

const RING_VERT = `
  varying vec2 vUv;
  varying vec3 vPosW;
  void main() {
    vUv = uv;
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const RING_FRAG = `
  varying vec2 vUv;
  varying vec3 vPosW;
  uniform vec3 uColor;
  uniform float uBright;
  uniform float uTime;
  uniform float uPhase;

  void main() {
    float x = abs(vUv.x * 2.0 - 1.0);
    float tubeRim = pow(x, 0.35);
    float pulse = 0.88 + 0.12 * sin(vUv.y * 24.0 + uTime * 1.8 + uPhase);
    float shimmer = 0.08 * sin(vUv.y * 48.0 - uTime * 3.2 + uPhase * 2.0);
    vec3 col = uColor * (0.22 + 0.78 * tubeRim) * pulse * (1.0 + shimmer);
    gl_FragColor = vec4(col * uBright, 1.0);
  }
`

const VEIL_VERT = `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const VEIL_FRAG = `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  uniform float uTime;
  uniform float uStrength;

  float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
  }

  void main() {
    vec3 n = normalize(vNormalW);
    vec3 v = normalize(cameraPosition - vPosW);
    float edge = pow(1.0 - max(dot(n, v), 0.0), 1.85);
    vec3 p = vPosW * 2.5;

    float s1 = hash31(floor(p * 12.0 + uTime * 0.08));
    float s2 = hash31(floor(p * 18.0 - uTime * 0.06));
    float stars = smoothstep(0.92, 1.0, s1) * smoothstep(0.88, 1.0, s2) * 0.55;

    float wisp = 0.5 + 0.5 * sin(dot(p, vec3(0.7, 1.1, 0.9)) + uTime * 0.7);
    wisp *= 0.5 + 0.5 * sin(p.y * 4.0 + uTime * 0.55);

    vec3 col = mix(vec3(0.55, 0.32, 0.1), vec3(1.0, 0.88, 0.55), edge * 0.9 + wisp * 0.12);
    col += vec3(1.0, 0.95, 0.8) * stars;
    float a = (0.06 + edge * 0.26 + stars * 0.35 + wisp * 0.06) * uStrength;
    gl_FragColor = vec4(col * a, a);
  }
`

const SHELL_VERT = `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vPosW = world.xyz;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const SHELL_FRAG = `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  uniform float uTime;
  void main() {
    vec3 n = normalize(vNormalW);
    vec3 v = normalize(cameraPosition - vPosW);
    float fr = pow(1.0 - max(dot(n, v), 0.0), 2.8);
    float pulse = 0.92 + 0.08 * sin(uTime * 1.4 + dot(vPosW, vec3(0.2, 0.15, 0.18)));
    vec3 col = vec3(1.0, 0.82, 0.52) * fr * pulse * 0.42;
    gl_FragColor = vec4(col, fr * 0.35);
  }
`

const GLOW_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const GLOW_FRAG = `
  varying vec2 vUv;
  uniform float uTime;
  uniform float uStrength;

  void main() {
    vec2 c = vUv - vec2(0.5);
    float r = length(c) * 2.0;
    float blob = exp(-r * r * 1.35);
    float scan = 0.65 + 0.35 * sin(vUv.y * 420.0 + uTime * 1.2);
    float slow = 0.85 + 0.15 * sin(vUv.y * 28.0 - uTime * 0.4);
    vec3 col = vec3(1.0, 0.72, 0.38) * blob * scan * slow;
    float a = blob * 0.55 * uStrength;
    gl_FragColor = vec4(col * a, a);
  }
`

export const Orb = forwardRef<Group, OrbProps>(({ energy, transform, lightEnabled = true }, ref) => {
  const groupRef = useRef<Group>(null)
  const ringRefs = useRef<Array<Mesh | null>>(Array(RING_COUNT).fill(null))
  const ringMatRefs = useRef<Array<ShaderMaterial | null>>(Array(RING_COUNT).fill(null))
  const coreRef = useRef<Mesh>(null)
  const coreMatRef = useRef<MeshBasicMaterial>(null)
  const veilMatRef = useRef<ShaderMaterial>(null)
  const shellMatRef = useRef<ShaderMaterial>(null)
  const glowMatRef = useRef<ShaderMaterial>(null)
  const rimMatRef = useRef<ShaderMaterial>(null)
  const lightRef = useRef<PointLight>(null)
  const lightEnabledRef = useRef(lightEnabled)
  
  useEffect(() => {
    lightEnabledRef.current = lightEnabled
  }, [lightEnabled])
  
  const smoothedEnergy = useRef(0)
  const peakEnergy = useRef(0)

  useFrame((state) => {
    const t = state.clock.getElapsedTime()

    const attack = 0.55
    const release = 0.18
    if (energy > smoothedEnergy.current) {
      smoothedEnergy.current += (energy - smoothedEnergy.current) * attack
    } else {
      smoothedEnergy.current += (energy - smoothedEnergy.current) * release
    }

    peakEnergy.current = Math.max(peakEnergy.current * 0.86, energy)
    const audioLevel = Math.min(
      1,
      Math.pow(smoothedEnergy.current * 0.65 + peakEnergy.current * 0.35, 0.75) * 1.15,
    )

    const innerAudio = Math.min(1, Math.pow(audioLevel, 0.82) * 1.1)

    const driftX =
      Math.sin(t * 1.4) * (0.15 + audioLevel * 0.62) +
      Math.sin(t * 3.1) * (0.05 + audioLevel * 0.16)
    const driftY =
      Math.sin(t * 1.9) * (0.07 + audioLevel * 0.36) +
      Math.sin(t * 5.2) * (0.02 + audioLevel * 0.08)
    const driftZ =
      Math.cos(t * 1.1) * (0.15 + audioLevel * 0.58) +
      Math.cos(t * 3.6) * (0.045 + audioLevel * 0.1)

    if (groupRef.current) {
      groupRef.current.position.lerp(
        new Vector3(
          transform.position[0] + driftX,
          transform.position[1] + driftY,
          transform.position[2] + driftZ,
        ),
        0.12,
      )
      groupRef.current.rotation.y += 0.005 + audioLevel * 0.04
      const pulse =
        1 + Math.sin(t * (2.7 + audioLevel * 5.4)) * (0.01 + audioLevel * 0.028) + audioLevel * 0.03
      groupRef.current.scale.setScalar(transform.scale * pulse)
      groupRef.current.updateMatrixWorld(true)
    }

    const bright = 1.15 + innerAudio * 1.35
    for (let i = 0; i < RING_COUNT; i += 1) {
      const ring = ringRefs.current[i]
      const mat = ringMatRefs.current[i]
      if (!ring || !mat) continue

      const phase = i * 1.09
      ring.position.set(
        Math.sin(t * 0.5 + phase) * 0.018,
        Math.cos(t * 0.44 + phase * 0.7) * 0.016,
        Math.sin(t * 0.58 + phase * 1.2) * 0.015,
      )

      ring.rotation.x = t * (0.52 + i * 0.06) + phase * 0.25
      ring.rotation.y = t * (0.41 + i * 0.05) + i * 0.72
      ring.rotation.z = t * (0.28 + i * 0.04) + phase * 0.12

      mat.uniforms.uBright.value = bright * (0.85 + i * 0.04) + Math.sin(t * 2.5 + phase) * 0.12
      mat.uniforms.uTime.value = t
    }

    if (coreRef.current && coreMatRef.current) {
      const s = 1 + Math.sin(t * 2.8) * (0.05 + innerAudio * 0.08) + innerAudio * 0.06
      coreRef.current.scale.setScalar(s)
      coreMatRef.current.color.setRGB(
        1,
        0.96 + innerAudio * 0.04,
        0.88 + innerAudio * 0.1,
      )
    }

    if (veilMatRef.current) {
      veilMatRef.current.uniforms.uTime.value = t
      veilMatRef.current.uniforms.uStrength.value = 0.78 + innerAudio * 0.45
    }
    if (shellMatRef.current) {
      shellMatRef.current.uniforms.uTime.value = t
    }
    if (glowMatRef.current) {
      glowMatRef.current.uniforms.uTime.value = t
      glowMatRef.current.uniforms.uStrength.value = 0.55 + innerAudio * 0.55 + audioLevel * 0.25
    }

    if (rimMatRef.current) {
      rimMatRef.current.uniforms.uTime.value = t
      rimMatRef.current.uniforms.audioLevel.value = audioLevel
    }
    if (lightRef.current) {
      if (lightEnabledRef.current) {
        lightRef.current.intensity = 22 + audioLevel * 58 + innerAudio * 20
        lightRef.current.distance = 6.5 + audioLevel * 5.5
      } else {
        lightRef.current.intensity = 0
      }
    }
  }, -1)

  return (
    <group
      ref={(node) => {
        groupRef.current = node
        if (typeof ref === 'function') {
          ref(node)
        } else if (ref) {
          ref.current = node
        }
      }}
      position={transform.position}
    >
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.58, 0]} renderOrder={2}>
        <circleGeometry args={[0.72, 64]} />
        <shaderMaterial
          ref={glowMatRef}
          transparent
          depthWrite={false}
          depthTest={false}
          blending={AdditiveBlending}
          toneMapped={false}
          side={DoubleSide}
          uniforms={{
            uTime: { value: 0 },
            uStrength: { value: 0.85 },
          }}
          vertexShader={GLOW_VERT}
          fragmentShader={GLOW_FRAG}
        />
      </mesh>

      <group renderOrder={4}>
        <mesh renderOrder={4}>
          <sphereGeometry args={[0.41, 40, 40]} />
          <shaderMaterial
            ref={veilMatRef}
            transparent
            depthWrite={false}
            depthTest={false}
            blending={AdditiveBlending}
            toneMapped={false}
            side={DoubleSide}
            uniforms={{
              uTime: { value: 0 },
              uStrength: { value: 0.9 },
            }}
            vertexShader={VEIL_VERT}
            fragmentShader={VEIL_FRAG}
          />
        </mesh>

        <mesh renderOrder={4}>
          <sphereGeometry args={[0.44, 32, 32]} />
          <shaderMaterial
            ref={shellMatRef}
            transparent
            depthWrite={false}
            depthTest={false}
            blending={AdditiveBlending}
            toneMapped={false}
            side={DoubleSide}
            uniforms={{ uTime: { value: 0 } }}
            vertexShader={SHELL_VERT}
            fragmentShader={SHELL_FRAG}
          />
        </mesh>

        {Array.from({ length: RING_COUNT }).map((_, i) => {
          const phase = i * 1.09
          const hue = 0.02 * i
          return (
            <mesh
              key={i}
              renderOrder={5}
              ref={(node) => {
                ringRefs.current[i] = node
              }}
            >
              <torusGeometry args={[ringMajor[i], ringTube[i], 24, 160]} />
              <shaderMaterial
                ref={(node) => {
                  ringMatRefs.current[i] = node
                }}
                transparent
                depthWrite={false}
                depthTest={false}
                blending={AdditiveBlending}
                toneMapped={false}
                side={DoubleSide}
                uniforms={{
                  uColor: { value: new Vector3(1, 0.96 - hue * 0.15, 0.82 - hue * 0.1) },
                  uBright: { value: 1.2 },
                  uTime: { value: 0 },
                  uPhase: { value: phase },
                }}
                vertexShader={RING_VERT}
                fragmentShader={RING_FRAG}
              />
            </mesh>
          )
        })}

        <mesh ref={coreRef} renderOrder={6}>
          <sphereGeometry args={[0.048, 24, 24]} />
          <meshBasicMaterial
            ref={coreMatRef}
            color="#fffcef"
            toneMapped={false}
            blending={AdditiveBlending}
            depthWrite={false}
            depthTest={false}
          />
        </mesh>
      </group>

      <mesh renderOrder={7}>
        <sphereGeometry args={[0.45, 128, 96]} />
        <meshPhysicalMaterial
          transmission={0.56}
          roughness={0.08}
          thickness={0.22}
          ior={1.1}
          transparent
          opacity={0.1}
          color="#f8f0e6"
          metalness={0.02}
          reflectivity={0.22}
          clearcoat={0.88}
          clearcoatRoughness={0.12}
          envMapIntensity={0.5}
          depthWrite={false}
          side={FrontSide}
        />
      </mesh>

      <mesh renderOrder={8}>
        <sphereGeometry args={[0.458, 96, 72]} />
        <shaderMaterial
          ref={rimMatRef}
          transparent
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
          uniforms={{
            uTime: { value: 0 },
            audioLevel: { value: 0 },
          }}
          vertexShader={`
            varying vec3 vPosW;
            varying vec3 vNormalW;
            void main() {
              vec4 world = modelMatrix * vec4(position, 1.0);
              vPosW = world.xyz;
              vNormalW = normalize(mat3(modelMatrix) * normal);
              gl_Position = projectionMatrix * viewMatrix * world;
            }
          `}
          fragmentShader={`
            varying vec3 vPosW;
            varying vec3 vNormalW;
            uniform float uTime;
            uniform float audioLevel;
            void main() {
              vec3 n = normalize(vNormalW);
              vec3 v = normalize(cameraPosition - vPosW);
              float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0);
              float a = clamp(audioLevel, 0.0, 1.0);
              float breathe = 0.88 + sin(uTime * (2.4 + a * 3.2)) * (0.04 + a * 0.05);
              float intensity = fresnel * (0.065 + a * 0.13) * breathe;
              vec3 rimColor = mix(vec3(0.95, 0.78, 0.45), vec3(1.0, 0.95, 0.82), 0.55);
              gl_FragColor = vec4(rimColor * intensity * 2.6, intensity * 0.9);
            }
          `}
        />
      </mesh>

      <pointLight ref={lightRef} intensity={14} color="#ffe8c8" distance={6.8} />
    </group>
  )
})

Orb.displayName = 'Orb'
