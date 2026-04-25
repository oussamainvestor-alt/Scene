import { OrbitControls, TransformControls, Environment } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import { Group, Vector3 } from 'three'
import { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { CameraCoordinates, SceneLayout, HdrType, RendererType, OrbLighting, GroundGrid } from '../../types'
import { FloatingScreen } from './FloatingScreen'
import { Orb } from './Orb'
import { ConcreteGround } from './Ground'

type EnvironmentSceneProps = {
  videoUrl: string | null
  orbEnergy: number
  orbLighting?: OrbLighting
  rendererType?: RendererType
  hdrType?: HdrType
  groundGrid?: GroundGrid
  onCanvasReady: (canvas: HTMLCanvasElement | null) => void
  layout: SceneLayout
  dragTarget: 'none' | 'orb' | 'screen' | 'ground'
  onLayoutChange: (next: SceneLayout) => void
  cameraCoordinates: CameraCoordinates
  onCameraCoordinatesChange: (next: CameraCoordinates) => void
}

type SceneRef = {
  getCoordinates: () => CameraCoordinates
  setCoordinates: (next: CameraCoordinates) => void
}

export type EnvironmentSceneHandle = SceneRef

export const EnvironmentScene = forwardRef<EnvironmentSceneHandle, EnvironmentSceneProps>(
  (
    {
      videoUrl,
      orbEnergy,
      orbLighting,
      rendererType,
      hdrType,
      groundGrid,
      onCanvasReady,
      layout,
      dragTarget,
      onLayoutChange,
      cameraCoordinates,
      onCameraCoordinatesChange,
    },
    ref,
  ) => {
    const controlsRef = useRef<OrbitControlsImpl>(null)
    const orbRef = useRef<Group>(null)
    const screenRef = useRef<Group>(null)
    const groundRef = useRef<Group>(null)
    const groupRef = useRef<Group>(null)

    useEffect(() => {
      const zoomFactor = 1.08
      const moveStep = 0.18
      const verticalStep = 0.18
      const upAxis = new Vector3(0, 1, 0)
      const forward = new Vector3()
      const right = new Vector3()
      const delta = new Vector3()
      let spacePressed = false

      const shouldIgnoreShortcut = (target: EventTarget | null) => {
        if (!(target instanceof HTMLElement)) return false
        const tag = target.tagName
        return (
          target.isContentEditable ||
          tag === 'INPUT' ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          !!target.closest('input, textarea, select, [contenteditable="true"]')
        )
      }

      const handleKeyDown = (event: KeyboardEvent) => {
        if (event.code === 'Space') {
          spacePressed = true
          return
        }

        if (shouldIgnoreShortcut(event.target)) return
        const controls = controlsRef.current
        if (!controls || !controls.enabled) return

        const key = event.key
        const isArrow = key === 'ArrowUp' || key === 'ArrowDown' || key === 'ArrowLeft' || key === 'ArrowRight'
        if (!isArrow) return

        event.preventDefault()

        if (event.ctrlKey && (key === 'ArrowUp' || key === 'ArrowDown')) {
          if (key === 'ArrowUp') {
            controls.dollyOut(zoomFactor)
          } else {
            controls.dollyIn(zoomFactor)
          }
          controls.update()
          emitCameraCoordinates()
          return
        }

        forward
          .subVectors(controls.target, controls.object.position)
          .setY(0)
        if (forward.lengthSq() < 1e-8) {
          forward.set(0, 0, -1)
        } else {
          forward.normalize()
        }
        right.crossVectors(forward, upAxis).normalize()
        delta.set(0, 0, 0)

        if (spacePressed && key === 'ArrowUp') delta.y += verticalStep
        if (spacePressed && key === 'ArrowDown') delta.y -= verticalStep
        if (!spacePressed && key === 'ArrowUp') delta.addScaledVector(forward, moveStep)
        if (!spacePressed && key === 'ArrowDown') delta.addScaledVector(forward, -moveStep)
        if (key === 'ArrowLeft') delta.addScaledVector(right, -moveStep)
        if (key === 'ArrowRight') delta.addScaledVector(right, moveStep)

        if (delta.lengthSq() === 0) return
        controls.object.position.add(delta)
        controls.target.add(delta)
        controls.update()
        emitCameraCoordinates()
      }

      const handleKeyUp = (event: KeyboardEvent) => {
        if (event.code === 'Space') {
          spacePressed = false
        }
      }

      window.addEventListener('keydown', handleKeyDown)
      window.addEventListener('keyup', handleKeyUp)
      return () => {
        window.removeEventListener('keydown', handleKeyDown)
        window.removeEventListener('keyup', handleKeyUp)
      }
    }, [])

    useImperativeHandle(
      ref,
      () => ({
        getCoordinates() {
          const controls = controlsRef.current
          if (!controls) {
            return {
              position: [4.2, 2.2, 4.4],
              target: [0, 1.2, 0],
              zoom: 1,
            }
          }
          return {
            position: [
              controls.object.position.x,
              controls.object.position.y,
              controls.object.position.z,
            ],
            target: [controls.target.x, controls.target.y, controls.target.z],
            zoom: controls.object.zoom,
          }
        },
        setCoordinates(next) {
          const controls = controlsRef.current
          if (!controls) {
            return
          }
          controls.object.position.set(...next.position)
          controls.target.set(...next.target)
          controls.object.zoom = next.zoom || 1
          controls.object.updateProjectionMatrix()
          controls.update()
        },
      }),
      [],
    )

    useEffect(() => {
      const controls = controlsRef.current
      if (!controls) return
      controls.object.position.set(...cameraCoordinates.position)
      controls.target.set(...cameraCoordinates.target)
      controls.object.zoom = cameraCoordinates.zoom || 1
      controls.object.updateProjectionMatrix()
      controls.update()
    }, [cameraCoordinates])

    const emitCameraCoordinates = () => {
      const controls = controlsRef.current
      if (!controls) return
      onCameraCoordinatesChange({
        position: [controls.object.position.x, controls.object.position.y, controls.object.position.z],
        target: [controls.target.x, controls.target.y, controls.target.z],
        zoom: controls.object.zoom,
      })
    }

    return (
      <Canvas
        shadows
        dpr={[1, 1.6]}
        gl={rendererType === 'webgpu' ? { powerPreference: 'high-performance', antialias: true } : undefined}
        camera={{ position: [4.2, 2.2, 4.4], fov: 40 }}
        onCreated={({ gl }) => {
          gl.setClearColor('#070a11')
          onCanvasReady(gl.domElement)
        }}
      >
        <fog attach="fog" args={['#060a11', 5, 18]} />
        <ambientLight intensity={0.08} color="#162133" />
        <hemisphereLight intensity={0.18} color="#132238" groundColor="#090d16" />
        <directionalLight
          castShadow
          intensity={0.6}
          color="#7a98c0"
          position={[2.5, 4.2, 1.8]}
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <spotLight
          castShadow
          position={[0, 5.8, 0.6]}
          angle={0.58}
          penumbra={0.9}
          intensity={7}
          color="#b5c9ef"
          distance={13}
        />
        <pointLight intensity={0.7} position={[-2.5, 1.4, -2]} color="#41618f" />

        {groundGrid === 1 ? (
          <group ref={groundRef}>
            <ConcreteGround scale={1} />
          </group>
        ) : groundGrid === 2 ? (
          <group ref={groundRef}>
            <ConcreteGround scale={1} position={[-2.5, 0, -2.5]} />
            <ConcreteGround scale={1} position={[2.5, 0, -2.5]} />
            <ConcreteGround scale={1} position={[-2.5, 0, 2.5]} />
            <ConcreteGround scale={1} position={[2.5, 0, 2.5]} />
          </group>
        ) : (
          <group>
            <ConcreteGround scale={1} position={[-3.75, 0, -3.75]} />
            <ConcreteGround scale={1} position={[-1.25, 0, -3.75]} />
            <ConcreteGround scale={1} position={[1.25, 0, -3.75]} />
            <ConcreteGround scale={1} position={[3.75, 0, -3.75]} />
            <ConcreteGround scale={1} position={[-3.75, 0, -1.25]} />
            <ConcreteGround scale={1} position={[-1.25, 0, -1.25]} />
            <ConcreteGround scale={1} position={[1.25, 0, -1.25]} />
            <ConcreteGround scale={1} position={[3.75, 0, -1.25]} />
            <ConcreteGround scale={1} position={[-3.75, 0, 1.25]} />
            <ConcreteGround scale={1} position={[-1.25, 0, 1.25]} />
            <ConcreteGround scale={1} position={[1.25, 0, 1.25]} />
            <ConcreteGround scale={1} position={[3.75, 0, 1.25]} />
            <ConcreteGround scale={1} position={[-3.75, 0, 3.75]} />
            <ConcreteGround scale={1} position={[-1.25, 0, 3.75]} />
            <ConcreteGround scale={1} position={[1.25, 0, 3.75]} />
            <ConcreteGround scale={1} position={[3.75, 0, 3.75]} />
          </group>
        )}
        <group ref={groupRef} rotation={[0, layout.groupRotation, 0]}>
          <Orb ref={orbRef} energy={orbEnergy} transform={layout.orb} lightEnabled={orbLighting} />

          {hdrType && hdrType.length > 0 && <Environment background={false} files={`/${hdrType}`} />}
          <FloatingScreen ref={screenRef} videoUrl={videoUrl} transform={layout.screen} />
        </group>

        {dragTarget === 'orb' && orbRef.current && (
          <TransformControls
            object={orbRef.current}
            mode="translate"
            onMouseDown={() => {
              if (controlsRef.current) controlsRef.current.enabled = false
            }}
            onMouseUp={() => {
              if (controlsRef.current) controlsRef.current.enabled = true
              if (!orbRef.current) return
              onLayoutChange({
                ...layout,
                orb: {
                  position: [
                    orbRef.current.position.x,
                    orbRef.current.position.y,
                    orbRef.current.position.z,
                  ],
                  scale: orbRef.current.scale.x,
                },
              })
            }}
          />
        )}
        {dragTarget === 'screen' && screenRef.current && (
          <TransformControls
            object={screenRef.current}
            mode="translate"
            onMouseDown={() => {
              if (controlsRef.current) controlsRef.current.enabled = false
            }}
            onMouseUp={() => {
              if (controlsRef.current) controlsRef.current.enabled = true
              if (!screenRef.current) return
              onLayoutChange({
                ...layout,
                screen: {
                  ...layout.screen,
                  position: [
                    screenRef.current.position.x,
                    screenRef.current.position.y,
                    screenRef.current.position.z,
                  ],
                  scale: screenRef.current.scale.x,
                },
              })
            }}
          />
        )}

        {dragTarget === 'ground' && groundRef.current && (
          <TransformControls
            object={groundRef.current}
            mode="translate"
            onMouseDown={() => {
              if (controlsRef.current) controlsRef.current.enabled = false
            }}
            onMouseUp={() => {
              if (controlsRef.current) controlsRef.current.enabled = true
            }}
          />
        )}

        <mesh position={[0, 3.8, -9]}>
          <planeGeometry args={[22, 12]} />
          <meshBasicMaterial color="#070b12" />
        </mesh>
        <OrbitControls
          ref={controlsRef}
          enablePan
          enableZoom
          enableDamping
          dampingFactor={0.06}
          onEnd={emitCameraCoordinates}
        />
      </Canvas>
    )
  },
)

EnvironmentScene.displayName = 'EnvironmentScene'
