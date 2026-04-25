import { useGLTF } from '@react-three/drei'

type ConcreteGroundProps = {
  scale?: number
  position?: [number, number, number]
}

export function ConcreteGround({ scale = 1, position = [0, 0, 0] }: ConcreteGroundProps) {
  const { scene } = useGLTF('/ground_concrete.glb')

  return (
    <primitive
      object={scene.clone()}
      scale={scale}
      position={position as any}
      receiveShadow
    />
  )
}
