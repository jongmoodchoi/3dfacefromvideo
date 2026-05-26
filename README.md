# 3dfacefromvideo
3D face reconstruction from video using multiview geometry.

## Run locally

1. Start a local static server from the repository root:
   ```bash
   python3 -m http.server 8000
   ```
2. Open `http://localhost:8000/index.html` in a modern desktop browser.
3. Allow camera access.

## Usage

1. Click **Start Camera**.
2. Click **Capture Sequence**.
3. Slowly turn your head left to right and back.
4. Click **Reconstruct 3D Face** to build and view the textured mesh.
5. Drag to rotate and use mouse wheel/touchpad to zoom.

## Pipeline implemented in the web app

- Facial landmark extraction with MediaPipe Face Mesh.
- Cross-frame keypoint correspondence by landmark index.
- Approximate head motion estimation (yaw, pitch, roll) from stable facial landmarks.
- Multi-view fusion by inverse-pose alignment of landmarks from captured views.
- Surface generation via Delaunay triangulation and texture projection from a near-frontal reference frame.
- Interactive 3D visualization with Three.js + OrbitControls.
