import { useFrame } from '@react-three/fiber'
import { forwardRef, useRef } from 'react'
import { Group, PointLight, ShaderMaterial } from 'three'

type OrbProps = {
  energy: number
  transform?: { position: [number, number, number]; scale: number }
  lightEnabled?: { ambient: number; fill: number; rim: number }
}

const ORB_VERT = /* glsl */ `
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorldPos = world.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const ORB_FRAG = /* glsl */ `
  uniform float uEnergy;
  uniform float uEmitStr;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;

  void main() {
    vec3 N = normalize(vWorldNormal);
    vec3 V = normalize(cameraPosition - vWorldPos);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float limb  = 1.0 - NdotV;

    float innerLight = pow(NdotV, 0.50);

    vec3 hotspot = vec3(1.00, 0.97, 0.92);
    vec3 cream   = vec3(1.00, 0.88, 0.70);
    vec3 peach   = vec3(0.94, 0.74, 0.50);
    vec3 darkRim = vec3(0.22, 0.16, 0.10);

    vec3 col = mix(peach, cream, innerLight);
    col = mix(col, hotspot, innerLight * innerLight * 0.55);

    col = mix(col, darkRim, pow(limb, 1.2) * 0.42);

    col += vec3(0.16, 0.08, 0.02) * uEnergy * NdotV;
    col *= uEmitStr;

    float alpha = 0.72 + pow(limb, 2.0) * 0.18;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), alpha);
  }
`

export const Orb = forwardRef<Group, OrbProps>(({ energy, transform, lightEnabled }, ref) => {
  const groupRef = useRef<Group>(null)
  const orbMatRef = useRef<ShaderMaterial>(null)
  const lightRef = useRef<PointLight>(null)

  const smEnergy = useRef(0)
  const smoothPos = useRef({ x: 0, y: 0.65, z: 0 })

  useFrame((state, dt) => {
    const t = state.clock.getElapsedTime()

    if (energy > smEnergy.current) {
      smEnergy.current += (energy - smEnergy.current) * Math.min(1, dt * 20)
    } else {
      smEnergy.current += (energy - smEnergy.current) * Math.min(1, dt * 4)
    }
    const e = smEnergy.current

    const breathSin = Math.sin(t * 1.8)
    const breathScale = 1.0 + breathSin * 0.015
    const audioScale = 1.0 + e * 0.35

    const g = groupRef.current
    if (g) {
      g.scale.setScalar((breathScale * audioScale) * (transform?.scale ?? 1))

      if (transform?.position) {
        g.position.x = transform.position[0] + smoothPos.current.x
        g.position.y = transform.position[1] + smoothPos.current.y
        g.position.z = transform.position[2] + smoothPos.current.z
      } else {
        const idleX = Math.sin(t * 0.3) * 0.1 + Math.sin(t * 0.5) * 0.05
        const idleZ = Math.cos(t * 0.25) * 0.08 + Math.cos(t * 0.4) * 0.04
        const idleY = Math.sin(t * 0.2) * 0.06

        const excite = e * 1.8
        const exciteX = Math.sin(t * 2.5) * 0.2 * excite
        const exciteZ = Math.cos(t * 2.0) * 0.15 * excite
        const exciteY = Math.abs(Math.sin(t * 3.5)) * 0.12 * excite

        const drift = 0.1 + e * 0.2
        const driftX = Math.sin(t * 0.15) * drift
        const driftZ = Math.cos(t * 0.12) * drift

        const targetX = idleX + exciteX * 0.6 + driftX
        const targetY = 0.65 + idleY + exciteY
        const targetZ = idleZ + exciteZ * 0.6 + driftZ

        const smooth = 2 + e * 3
        smoothPos.current.x += (targetX - smoothPos.current.x) * dt * smooth
        smoothPos.current.y += (targetY - smoothPos.current.y) * dt * smooth
        smoothPos.current.z += (targetZ - smoothPos.current.z) * dt * smooth

        const velX = (targetX - smoothPos.current.x) * smooth
        const velZ = (targetZ - smoothPos.current.z) * smooth

        g.position.x = smoothPos.current.x
        g.position.y = smoothPos.current.y
        g.position.z = smoothPos.current.z

        const lean = 1.0 + e * 0.5
        g.rotation.z = -velX * lean * 0.3 - Math.sin(t * 2.5) * e * 0.3
        g.rotation.x = velZ * lean * 0.25 - Math.cos(t * 2.0) * e * 0.25
        g.rotation.y += (t * 0.015 + e * 0.1 - g.rotation.y) * dt * 3
      }

      if (lightEnabled) {
        if (lightRef.current) {
          lightRef.current.intensity = (6 + e * 18) * lightEnabled.ambient
        }
      }
    }

    const emitStr = (1.10 + Math.sin(t * 1.8) * 0.08) * (1.0 + e * 0.7)

    if (orbMatRef.current) {
      orbMatRef.current.uniforms.uEnergy.value = e
      orbMatRef.current.uniforms.uEmitStr.value = emitStr
    }
    if (lightRef.current) {
      if (!lightEnabled) {
        lightRef.current.intensity = 6 + e * 18
      }
    }
  })

  return (
    <group
      ref={(node) => {
        groupRef.current = node
        if (typeof ref === 'function') ref(node)
        else if (ref) ref.current = node
      }}
    >
      <mesh>
        <sphereGeometry args={[0.44, 64, 48]} />
        <shaderMaterial
          ref={orbMatRef}
          transparent
          depthWrite={true}
          toneMapped={false}
          uniforms={{
            uEnergy: { value: 0 },
            uEmitStr: { value: 1.1 },
          }}
          vertexShader={ORB_VERT}
          fragmentShader={ORB_FRAG}
        />
      </mesh>

      <pointLight ref={lightRef} color="#ffaa44" intensity={6} distance={5} decay={2} />
    </group>
  )
})

Orb.displayName = 'Orb'