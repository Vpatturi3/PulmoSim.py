import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'

type STLViewerProps = {
  src: string
  airwaySrc?: string | null
  seeInside?: boolean
  airwaysVisible?: boolean
  airflowVisible?: boolean
  onToggleInside?: () => void
  onLoaded?(meta: { vertices: number; faces: number; bbox: [number, number, number][] }): void
}

function STLContent({ src, airwaySrc, seeInside = false, airwaysVisible = false, airflowVisible = false, onToggleInside, onLoaded }: STLViewerProps) {
  const { gl, camera, size, scene } = useThree()
  const geometry = useLoader(STLLoader as any, src) as THREE.BufferGeometry
  const meshRef = useRef<THREE.Mesh>(null!)
  const bboxHelperRef = useRef<THREE.Box3Helper | null>(null)
  const downRef = useRef<{ x: number; y: number; t: number } | null>(null)

  // Materials
  const lungsMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: new THREE.Color('#77bcd7'), roughness: 0.55, metalness: 0.1, transparent: true, opacity: seeInside ? 0.28 : 1 }), [seeInside])
  const airwayMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: new THREE.Color('#ff4d4d'), roughness: 0.6, metalness: 0.05, transparent: true, opacity: 0.95 }), [])

  useEffect(() => {
    gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    // Make canvas fully transparent (no dark square)
    ;(gl as THREE.WebGLRenderer).setClearColor(0x000000, 0)
  }, [gl])

  // Configure mesh once geometry is loaded
  useEffect(() => {
    geometry.computeVertexNormals()
    geometry.center()

    // Fit camera to object and colorize vertices
    const tempMesh = new THREE.Mesh(geometry)
    const box = new THREE.Box3().setFromObject(tempMesh)
    const sizeVec = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(sizeVec.x, sizeVec.y, sizeVec.z)
    const fov = (camera as THREE.PerspectiveCamera).fov * (Math.PI / 180)
    const camZ = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 2.6
    camera.position.set(0, 0, camZ)
    ;(camera as THREE.PerspectiveCamera).near = maxDim / 100
    ;(camera as THREE.PerspectiveCamera).far = maxDim * 10
    camera.updateProjectionMatrix()

    // Keep a uniform color; no vertex coloring applied

    onLoaded?.({
      vertices: geometry.attributes.position.count,
      faces: geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3,
      bbox: [
        [box.min.x, box.min.y, box.min.z],
        [box.max.x, box.max.y, box.max.z],
      ] as any,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, camera])

  // Keep background transparent at all times
  useEffect(() => {
    ;(gl as THREE.WebGLRenderer).setClearColor(0x000000, 0)
  }, [gl])

  useFrame(() => {
    // nothing animated
  })


  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[100, 100, 100]} intensity={0.6} />
      <group
        scale={[1.5, 1.5, 1.5]}
        rotation={[Math.PI, 0, 0]}
        onPointerDown={(e: any) => { downRef.current = { x: e.clientX, y: e.clientY, t: performance.now() } }}
        onPointerUp={(e: any) => {
          if (!downRef.current) return
          const dx = e.clientX - downRef.current.x
          const dy = e.clientY - downRef.current.y
          const dt = performance.now() - downRef.current.t
          downRef.current = null
          if (Math.hypot(dx, dy) < 5 && dt < 300) {
            onToggleInside?.()
          }
        }}
      >
        <mesh ref={meshRef} geometry={geometry} material={lungsMaterial} />
        {airwaysVisible && airwaySrc && (
          <AirwayMesh url={airwaySrc} material={airwayMaterial} />
        )}
        {airflowVisible && (
          <AirflowParticles boundsGeometry={geometry} color={new THREE.Color('#ff6b6b')} />
        )}
      </group>
      <OrbitControls makeDefault enableDamping />
    </>
  )
}

function AirflowParticles({ boundsGeometry, color }: { boundsGeometry: THREE.BufferGeometry, color: THREE.Color }) {
  const groupRef = useRef<THREE.Group>(null)
  const num = 200
  const positions = useMemo(() => Array.from({ length: num }, () => new THREE.Vector3(
    (Math.random() - 0.5) * 1.2,
    Math.random() * 1.2,
    (Math.random() - 0.5) * 1.2
  )), [])

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    positions.forEach((p, i) => {
      p.y -= 0.006 + (Math.sin(t + i) * 0.002)
      p.x += Math.sin(t * 1.4 + i) * 0.0015
      p.z += Math.cos(t * 1.2 + i) * 0.001
      if (p.y < -0.6) {
        p.y = 0.6
      }
    })
    if (groupRef.current) {
      groupRef.current.children.forEach((child, i) => {
        child.position.copy(positions[i])
      })
    }
  })

  return (
    <group ref={groupRef}>
      {positions.map((p, i) => (
        <mesh key={i} position={p.toArray() as any}>
          <sphereGeometry args={[0.006, 8, 8]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} transparent opacity={0.9} />
        </mesh>
      ))}
    </group>
  )
}

function AirwayMesh({ url, material }: { url: string, material: THREE.Material }) {
  const geom = useLoader(STLLoader as any, url) as THREE.BufferGeometry
  return <mesh geometry={geom} material={material} />
}

export default function STLViewer(props: STLViewerProps) {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Canvas camera={{ position: [0, 0, 300], near: 0.1, far: 5000 }} gl={{ alpha: true }} style={{ background: 'transparent' }}>
        {/* @ts-ignore custom loader is attached via global THREE */}
        <STLContent {...props} />
      </Canvas>
    </div>
  )
}


