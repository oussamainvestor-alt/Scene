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
  uniform float uCurveTop;
  uniform float uCurveBottom;
  uniform float uCurveLeft;
  uniform float uCurveRight;
  void main() {
    vUv = uv;
    vec3 warped = position;
    float nx = uv.x * 2.0 - 1.0;
    float ny = uv.y * 2.0 - 1.0;
    float topWeight    = smoothstep(0.0, 1.0, uv.y)       * (1.0 - nx * nx);
    float bottomWeight = smoothstep(0.0, 1.0, 1.0 - uv.y) * (1.0 - nx * nx);
    float leftWeight   = smoothstep(0.0, 1.0, 1.0 - uv.x) * (1.0 - ny * ny);
    float rightWeight  = smoothstep(0.0, 1.0, uv.x)        * (1.0 - ny * ny);
    warped.y += ((uCurveTop * topWeight)   - (uCurveBottom * bottomWeight)) * 1.8;
    warped.x += ((uCurveRight * rightWeight) - (uCurveLeft * leftWeight))   * 1.8;
    float depthCurve = (uCurveTop * topWeight) + (uCurveBottom * bottomWeight)
                     + (uCurveLeft * leftWeight) + (uCurveRight * rightWeight);
    warped.z += depthCurve * 0.9;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(warped, 1.0);
  }
`

const FRAGMENT_SHADER = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uVideoTex;
  uniform float uVideoAspect;
  uniform float uScreenAspect;
  uniform float uHasVideo;
  uniform float uRadius;
  uniform vec2 uSize;
  uniform vec3 uBackground;

  float roundedRectSdf(vec2 p, vec2 halfSize, float radius) {
    vec2 q = abs(p) - (halfSize - vec2(radius));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
  }

  void main() {
    vec2 centered = (vUv - 0.5) * uSize;
    float d = roundedRectSdf(centered, 0.5 * uSize, uRadius);
    if (d > 0.0) discard;

    vec3 color = uBackground;

    if (uHasVideo > 0.5) {
      vec2 sampleUv = vUv;
      bool inside = true;
      if (uVideoAspect > uScreenAspect) {
        float h = uScreenAspect / uVideoAspect;
        float yMin = 0.5 - h * 0.5;
        float yMax = 0.5 + h * 0.5;
        inside = vUv.y >= yMin && vUv.y <= yMax;
        sampleUv = vec2(vUv.x, (vUv.y - yMin) / h);
      } else {
        float w = uVideoAspect / uScreenAspect;
        float xMin = 0.5 - w * 0.5;
        float xMax = 0.5 + w * 0.5;
        inside = vUv.x >= xMin && vUv.x <= xMax;
        sampleUv = vec2((vUv.x - xMin) / w, vUv.y);
      }
      if (inside) {
        color = texture2D(uVideoTex, sampleUv).rgb;
      }
    }

    gl_FragColor = vec4(color, 1.0);
  }
`

export const FloatingScreen = forwardRef<Group, FloatingScreenProps>(({ videoUrl, transform }, ref) => {
  const planeWidth = 2.7
  const [ratioW, ratioH] = transform.aspectRatio
  const screenAspect = Math.max(1, ratioW || 1) / Math.max(1, ratioH || 1)
  const planeHeight = planeWidth / screenAspect
  const safeRadius = Math.min(Math.max(0, transform.borderRadius), Math.min(planeWidth, planeHeight) * 0.45)
  const { top, bottom, left, right } = transform.edgeCurve

  const [videoAspect, setVideoAspect] = useState(16 / 9)
  const lastPlayAttemptMs = useRef(0)

  const videoEl = useMemo(() => document.createElement('video'), [])

  const videoTexture = useMemo(() => {
    const t = new VideoTexture(videoEl)
    t.wrapS = ClampToEdgeWrapping
    t.wrapT = ClampToEdgeWrapping
    return t
  }, [videoEl])

  // Create the material imperatively — we own and mutate its uniforms directly
  const material = useMemo(() => new ShaderMaterial({
    uniforms: {
      uVideoTex:     { value: videoTexture },
      uVideoAspect:  { value: 16 / 9 },
      uScreenAspect: { value: screenAspect },
      uHasVideo:     { value: 0 },
      uRadius:       { value: safeRadius },
      uSize:         { value: new Vector2(planeWidth, planeHeight) },
      uBackground:   { value: new Vector3(0.03, 0.04, 0.07) },
      uCurveTop:     { value: top },
      uCurveBottom:  { value: bottom },
      uCurveLeft:    { value: left },
      uCurveRight:   { value: right },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    toneMapped: false,
  }), [videoTexture]) // eslint-disable-line react-hooks/exhaustive-deps

  // Attach video element to the DOM so the browser decodes frames reliably
  useEffect(() => {
    const el = videoEl
    el.style.cssText = 'position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none'
    document.body.appendChild(el)
    return () => {
      if (el.parentNode) el.parentNode.removeChild(el)
    }
  }, [videoEl])

  // Video element setup — runs whenever the URL changes
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

  // Update all uniforms and drive the texture every frame, right before Three.js renders
  useFrame(({ clock }) => {
    const u = material.uniforms
    u.uHasVideo.value     = videoUrl ? 1 : 0
    u.uVideoAspect.value  = videoAspect
    u.uScreenAspect.value = screenAspect
    u.uRadius.value       = safeRadius
    u.uSize.value.set(planeWidth, planeHeight)
    u.uCurveTop.value     = top
    u.uCurveBottom.value  = bottom
    u.uCurveLeft.value    = left
    u.uCurveRight.value   = right

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

  // Clean up the video element on unmount. We intentionally skip disposing the
  // ShaderMaterial and VideoTexture here because React StrictMode reuses useMemo
  // instances across its simulated unmount/remount cycle — disposing them in the
  // cleanup would break Three.js's WebGL texture and program tracking on remount.
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
