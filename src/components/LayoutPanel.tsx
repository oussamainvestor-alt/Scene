import { forwardRef, useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { ClampToEdgeWrapping, Group, ShaderMaterial, Vector2, Vector3, VideoTexture } from 'three'
import type { ScreenTransform } from '../../types'

type FloatingScreenProps = {
  videoUrl: string | null
  transform: ScreenTransform
}

const VERTEX_SHADER = `
  varying vec2 vUv;
  varying vec3 vLocalPos;
  uniform float uCurveTop;
  uniform float uCurveBottom;
  uniform float uCurveLeft;
  uniform float uCurveRight;

  void main() {
    vUv = uv;
    vLocalPos = position;

    vec3 warped = position;
    float nx = uv.x * 2.0 - 1.0;
    float ny = uv.y * 2.0 - 1.0;

    float topWeight    = smoothstep(0.0, 1.0, uv.y)       * (1.0 - nx * nx);
    float bottomWeight = smoothstep(0.0, 1.0, 1.0 - uv.y) * (1.0 - nx * nx);
    float leftWeight   = smoothstep(0.0, 1.0, 1.0 - uv.x) * (1.0 - ny * ny);
    float rightWeight  = smoothstep(0.0, 1.0, uv.x)       * (1.0 - ny * ny);

    warped.y += ((uCurveTop * topWeight)    - (uCurveBottom * bottomWeight)) * 1.8;
    warped.x += ((uCurveRight * rightWeight) - (uCurveLeft * leftWeight))    * 1.8;

    float depthCurve = (uCurveTop * topWeight) + (uCurveBottom * bottomWeight)
                     + (uCurveLeft * leftWeight) + (uCurveRight * rightWeight);
    warped.z += depthCurve * 0.9;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(warped, 1.0);
  }
`

const FRAGMENT_SHADER = `
  precision highp float;
  varying vec2 vUv;
  varying vec3 vLocalPos;
  uniform sampler2D uVideoTex;
  uniform float uVideoAspect;
  uniform float uScreenAspect;
  uniform float uHasVideo;
  uniform float uRadius;
  uniform vec2 uSize;
  uniform vec3 uBackground;
  uniform float uContentScale;

  float roundedRectSdf(vec2 p, vec2 halfSize, float radius) {
    vec2 q = abs(p) - (halfSize - vec2(radius));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
  }

  void main() {
    // Compute flat UVs from the undeformed local position.
    // This ensures the video stays flat even when the mesh curves.
    vec2 flatUv = vec2(vLocalPos.x / uSize.x + 0.5, vLocalPos.y / uSize.y + 0.5);

    // Rounded-rectangle clip (screen border)
    vec2 centered = (flatUv - 0.5) * uSize;
    float d = roundedRectSdf(centered, 0.5 * uSize, uRadius);
    if (d > 0.0) discard;

    vec3 color = uBackground;

    if (uHasVideo > 0.5) {
      // Aspect-ratio-correct video size in UV space
      float videoW, videoH;
      if (uVideoAspect > uScreenAspect) {
        videoW = 1.0;
        videoH = uScreenAspect / uVideoAspect;
      } else {
        videoW = uVideoAspect / uScreenAspect;
        videoH = 1.0;
      }

      // Apply content scale (zoom in/out)
      videoW *= uContentScale;
      videoH *= uContentScale;

      float xMin = 0.5 - videoW * 0.5;
      float yMin = 0.5 - videoH * 0.5;
      float xMax = 0.5 + videoW * 0.5;
      float yMax = 0.5 + videoH * 0.5;

      if (flatUv.x >= xMin && flatUv.x <= xMax && flatUv.y >= yMin && flatUv.y <= yMax) {
        vec2 texUv = vec2(
          (flatUv.x - xMin) / videoW,
          (flatUv.y - yMin) / videoH
        );
        color = texture2D(uVideoTex, texUv).rgb;
      }
    }

    gl_FragColor = vec4(color, 1.0);
  }
`

export const FloatingScreen = forwardRef(({ videoUrl, transform }, ref) => {
  const planeWidth = 2.7
  const [ratioW, ratioH] = transform.aspectRatio
  const screenAspect = Math.max(1, ratioW || 1) / Math.max(1, ratioH || 1)
  const planeHeight = planeWidth / screenAspect
  const safeRadius = Math.min(Math.max(0, transform.borderRadius), Math.min(planeWidth, planeHeight) * 0.45)
  const { top, bottom, left, right } = transform.edgeCurve
  const contentScale = transform.contentScale ?? 1.0

  const [videoAspect, setVideoAspect] = useState(16 / 9)
  const lastPlayAttemptMs = useRef(0)

  const videoEl = useMemo(() => document.createElement('video'), [])

  const videoTexture = useMemo(() => {
    const t = new VideoTexture(videoEl)
    t.wrapS = ClampToEdgeWrapping
    t.wrapT = ClampToEdgeWrapping
    return t
  }, [videoEl])

  const material = useMemo(() => new ShaderMaterial({
    uniforms: {
      uVideoTex: { value: videoTexture },
      uVideoAspect: { value: 16 / 9 },
      uScreenAspect: { value: screenAspect },
      uHasVideo: { value: 0 },
      uRadius: { value: safeRadius },
      uSize: { value: new Vector2(planeWidth, planeHeight) },
      uBackground: { value: new Vector3(0.03, 0.04, 0.07) },
      uCurveTop: { value: top },
      uCurveBottom: { value: bottom },
      uCurveLeft: { value: left },
      uCurveRight: { value: right },
      uContentScale: { value: contentScale },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    toneMapped: false,
  }), [videoTexture]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const el = videoEl
    el.style.cssText = 'position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none'
    document.body.appendChild(el)
    return () => {
      if (el.parentNode) el.parentNode.removeChild(el)
    }
  }, [videoEl])

  useEffect(() => {
    const attemptPlay = () => { videoEl.play().catch(() => undefined) }

    if (!videoUrl) {
      videoEl.pause()
      videoEl.removeAttribute('src')
      videoEl.load()
      return
    }

    videoEl.src = videoUrl
    videoEl.loop = true
    videoEl.muted = true
    videoEl.autoplay = true
    videoEl.preload = 'auto'
    videoEl.playsInline = true

    const onMeta = () => {
      if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
        setVideoAspect(videoEl.videoWidth / videoEl.videoHeight)
      }
    }
    const onCanPlay = () => {
      attemptPlay()
      videoTexture.needsUpdate = true
    }

    videoEl.addEventListener('loadedmetadata', onMeta)
    videoEl.addEventListener('canplay', onCanPlay)
    window.addEventListener('pointerdown', attemptPlay, { once: true })
    videoEl.load()
    attemptPlay()

    return () => {
      videoEl.removeEventListener('loadedmetadata', onMeta)
      videoEl.removeEventListener('canplay', onCanPlay)
      window.removeEventListener('pointerdown', attemptPlay)
    }
  }, [videoEl, videoUrl, videoTexture])

  useFrame(({ clock }) => {
    const u = material.uniforms
    u.uHasVideo.value = videoUrl ? 1 : 0
    u.uVideoAspect.value = videoAspect
    u.uScreenAspect.value = screenAspect
    u.uRadius.value = safeRadius
    u.uSize.value.set(planeWidth, planeHeight)
    u.uCurveTop.value = top
    u.uCurveBottom.value = bottom
    u.uCurveLeft.value = left
    u.uCurveRight.value = right
    u.uContentScale.value = contentScale

    if (!videoUrl) return

    const now = clock.elapsedTime * 1000
    if (videoEl.paused && now - lastPlayAttemptMs.current > 800) {
      lastPlayAttemptMs.current = now
      videoEl.play().catch(() => undefined)
    }

    if (videoEl.readyState >= 2) {
      videoTexture.needsUpdate = true
    }
  })

  useEffect(() => {
    return () => {
      videoEl.pause()
      videoEl.removeAttribute('src')
      videoEl.load()
    }
  }, [videoEl])

  return (
    <group
      ref={(node) => {
        if (typeof ref === 'function') ref(node)
        else if (ref) ref.current = node
      }}
      position={transform.position}
      scale={[transform.scale, transform.scale, transform.scale]}
    >
      <mesh>
        <planeGeometry args={[planeWidth, planeHeight, 64, 64]} />
        <primitive object={material} attach="material" />
      </mesh>
    </group>
  )
})

FloatingScreen.displayName = 'FloatingScreen'