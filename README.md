# Yoga Pose Correction System

A real-time yoga coaching application that watches you through a webcam, works out the angles of your
joints, and compares them against a reference version of the pose you selected. When a joint is out of
alignment the system tells you exactly what to change ("raise your left arm higher", "bend your right
knee more") both on screen and out loud through text-to-speech. Holding a pose correctly for two
seconds is treated as a completed repetition.

The reference pose for a comparison can come from one of two sources: a hand-authored table of target
joint angles built into the code, or angles extracted automatically from a reference photograph. This
hybrid approach means new poses can be added simply by dropping an image into `reference_poses/`.

## How it works

1. `PoseDetector` runs MediaPipe Pose on each webcam frame and returns 33 body landmarks.
2. `AngleCalculator` converts those landmarks into joint angles using the dot product between the two
   bone vectors that meet at each joint. It covers elbows, shoulders, knees and torso in `standard`
   mode, and additionally hips and ankles in `detailed` mode.
3. `PoseDatabase` supplies the target angles for the selected pose, either from the built-in
   `HARDCODED_POSES` table or by running MediaPipe over a reference image. Angles derived from images
   are cached in `reference_poses/pose_cache.json` so the extraction only happens once.
4. `PoseComparator` scores every joint against its target. Inside the tolerance band the score falls
   off linearly to zero at the tolerance limit; beyond it the score decays against a 180 degree span.
   The mean joint score is the overall similarity, and a pose counts as correct at 70 percent or above.
5. `FeedbackEngine` draws the score, per-joint angles and the highest-priority correction messages onto
   the frame, and speaks one correction at a time via `pyttsx3` with a configurable interval between
   utterances so the voice does not talk over itself.
6. `PoseVisualizer` provides the optional heatmap skeleton, where each limb is coloured green, yellow,
   orange or red according to that joint's score, plus a landmark-confidence overlay.

## Features

- Real-time skeleton tracking from any connected webcam, with automatic camera detection when more
  than one device is present.
- Four built-in poses with hand-tuned target angles: Tadasana (Mountain, arms up), Virabhadrasana II
  (Warrior 2), Vrksasana (Tree pose) and a Standing Side Bend.
- Additional poses can be added by placing a `.jpg`, `.jpeg` or `.png` in `reference_poses/`; the
  target angles are extracted from the image the first time the pose is selected.
- Colour-coded heatmap skeleton showing which limbs are correct and which are not.
- Landmark confidence overlay driven by MediaPipe visibility values.
- Spoken corrections and a spoken success message, prioritised so the worst joint is mentioned first.
- Reference image picture-in-picture overlay, digital zoom, and an on-screen FPS counter.
- Interactive pose selection menu with a live camera preview, with a text-based fallback when no
  camera is available.
- Frame skipping (every other frame is processed) and downscaling to 640 px wide to keep the loop
  responsive.

## Controls

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Quit |
| `P` | Change pose |
| `R` | Toggle the reference image overlay |
| `H` | Toggle the heatmap skeleton |
| `C` | Toggle the confidence overlay |
| `+` / `-` | Zoom in / out |
| `0` | Reset zoom |

In the selection menu, `W`/`S` navigate, `Enter` or `Space` selects, and `Q` or `Esc` cancels.

## Tech stack

- Python
- OpenCV (`opencv-python`) for capture, drawing and the UI windows
- MediaPipe Pose for landmark detection
- NumPy for the vector and angle maths
- pyttsx3 for offline text-to-speech

## Project structure

```
Yoga-Pose/
├── main.py                 # Entry point; camera loop, key handling, orchestration
├── pose_detector.py        # MediaPipe Pose wrapper
├── angle_calculator.py     # Joint angle extraction and similarity scoring
├── pose_database.py        # Built-in pose definitions, image-derived poses, caching
├── pose_comparator.py      # Per-joint scoring and correction messages
├── feedback_engine.py      # On-screen overlays and text-to-speech
├── pose_selector.py        # Interactive pose selection menu
├── pose_visualizer.py      # Heatmap skeleton and confidence visualisation
├── requirements.txt
└── reference_poses/
    ├── TADASANA_Mountain_Arms_Up.jpg
    ├── Virabhadrasana_Warrior_2.jpg
    ├── Vrksasana_Tree_Pose.jpg
    ├── Standing_Side_Bend_Pose.jpg
    ├── Moutain_Pose.png
    └── pose_cache.json     # Generated at runtime
```

## Getting started

A webcam is required. The pinned dependency versions target Python 3.9.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

On start-up the program scans camera indices 0 to 4, asks which camera to use if more than one is
found, then opens the pose selection menu. Pick a pose and stand far enough back that your whole body
is in frame.

To add your own pose, save a clear full-body photograph of it into `reference_poses/` using the pose
name as the filename. It will appear in the selection menu the next time the program starts.

## Author

Mohammad Salhab
