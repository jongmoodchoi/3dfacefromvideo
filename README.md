# 3dfacefromvideo

Browser-based 3D face reconstruction from camera video using **classical computer vision** techniques.

## What is implemented

This repository now contains a deployable static web app that:

1. Opens the laptop/web camera in the browser.
2. Tracks a sequence of facial landmarks using **CLM (Constrained Local Model)** tracking.
3. Reconstructs a 3D face shape from tracked 2D landmarks using **Tomasi-Kanade factorization** (SVD-based structure from motion).
4. Displays an interactive 3D face model in WebGL with rotate/zoom controls.

## Run locally

Because camera APIs require a secure context, run via `localhost`:

```bash
cd /tmp/workspace/jongmoodchoi/3dfacefromvideo
python3 -m http.server 8080
```

Open `http://localhost:8080` in a browser.

## Deploy on GCP (App Engine)

The project includes `app.yaml` for static hosting on Google App Engine.

```bash
gcloud config set project <YOUR_GCP_PROJECT_ID>
gcloud app create --region=<YOUR_REGION>   # first time only
gcloud app deploy
```

After deployment, open:

```text
https://<YOUR_GCP_PROJECT_ID>.appspot.com
```

## Usage flow

1. Click **Start Camera**.
2. Click **Capture Sequence** and slowly rotate your face for parallax.
3. Click **Reconstruct 3D**.
4. Rotate/zoom the model in the right viewer panel.
