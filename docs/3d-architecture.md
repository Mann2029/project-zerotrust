# AuraTrust Sentinel — 3D WebGL Visualization Architecture

## 1. Three.js Core Pipeline Specification

The 3D Sentinel Core is implemented using **Three.js 0.186.0** inside `frontend/src/components/3d/SentinelScene.jsx`.

```text
┌─────────────────────────────────────────────────────────────┐
│                       Three.js Scene                        │
├─────────────────┬───────────────────┬───────────────────────┤
│ PerspectiveCam  │ Directional Light │ WebGLRenderer         │
│ (FOV 45, D:5.2) │ (Cyan, CastShadow)│ (ACESFilmic, Damping) │
├─────────────────┼───────────────────┼───────────────────────┤
│ Ambient Light   │ Point Lights      │ Grid Helper &         │
│ (0x0a192f)      │ (Fill & Rim Cyan) │ Holographic Particles │
└─────────────────┴───────────────────┴───────────────────────┘
```

---

## 2. 3D Model Asset Specifications

| Attribute | Specification |
| :--- | :--- |
| **File Format** | Binary GLTF 2.0 (`.glb`) |
| **Asset Location** | `frontend/public/models/sentinel_shield.glb` (served at `/app/models/sentinel_shield.glb`) |
| **File Size** | 13,872 bytes (13.54 KB) |
| **Geometry** | 456 vertices, 152 indexed triangles |
| **Components** | Outer hexagonal security frame, inner dual-axis gimbal rings, central holographic core |
| **PBR Material** | Metallic: 0.85, Roughness: 0.20, BaseColor: `[0.05, 0.45, 0.85, 0.9]`, Emissive: `[0.08, 0.65, 0.95]` |

---

## 3. Performance Controller & Dynamic Tiering

The animation loop calculates actual frame times and executes dynamic hardware tiering:

- **High Tier ($FPS \ge 50$)**: Pixel ratio clamped to $\min(\text{devicePixelRatio}, 1.5)$, PCF soft shadow maps enabled, ACESFilmic tone mapping.
- **Low Tier ($FPS < 35$)**: Pixel ratio reduced to $1.0$, antialiasing load minimized.
- **Fail-Safe Procedural Fallback**: If the network fails to load the `.glb` binary asset, `SentinelScene` catches the exception and dynamically synthesizes a procedural composite geometry (Torus outer ring + Octahedron core) with zero interruption or layout shift.

---

## 4. Memory Lifecycle & Explicit Teardown

To prevent GPU VRAM leaks during portal navigation:
1. `cancelAnimationFrame(animationFrameId)` cancels the rendering loop.
2. The scene graph is traversed recursively:
   ```javascript
   scene.traverse((object) => {
     if (object.geometry) object.geometry.dispose();
     if (object.material) {
       if (Array.isArray(object.material)) object.material.forEach((m) => m.dispose());
       else object.material.dispose();
     }
   });
   ```
3. Particle buffer attributes and WebGLRenderer contexts are explicitly released via `renderer.dispose()`.
