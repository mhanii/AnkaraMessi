# AnkaraMessi: an AI football commentator

Give it a clip of a real match. It returns the same clip with spoken commentary, captions,
team-coloured markers under the players and a top-down mini-map.

It runs on a laptop CPU. No GPU and no paid API are needed.

## What it does

1. Finds the players, referee and ball in each frame, and tracks the players.
2. Works out where the camera is looking, so every player gets a position on the pitch in metres.
3. Splits players into two teams by shirt colour and spots the goalkeepers.
4. Recognises events: corner, penalty, kick-off, goal kick, free kick, shot, save, goal,
   pass, long pass, turnover, keeper throw, sustained pressure, replay.
5. Decides which events deserve a line and when, writes the line, and speaks it.
6. Renders everything back onto the video.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
./tools/fetch_models.sh                      # about 450 MB into ./models

python -m commentator run my_clip.mp4 --out out/my_clip
python -m commentator run my_clip.mp4 --out out/my_clip --teams "Ankara,Istanbul"
```

Outputs in `out/my_clip/`: `commentated.mp4`, `commentary.txt`, `commentary.wav`,
`events.json`, `summary.json`.

Expect about 6 minutes of processing per minute of video on 4 CPU cores. Use wide broadcast
footage from the main camera; close-ups are skipped.

## Results

Measured on SoccerNet GSR 2024: real 30-second broadcast clips with ground truth.

**Event rules, given perfect positions:** 29 of 33 labelled actions found on the clips used
for development, 35 of 40 on a second set. Before any tuning on that second set the score
was 30 of 40, which is the fair estimate for new footage.

**From raw video, end to end** (four clips):

| Clip | Labelled action | Found from video |
|---|---|---|
| SNGS-116 | corner at 6.3 s | corner kick at 6.3 s |
| SNGS-144 | penalty at 6.6 s | penalty at 6.3 s |
| SNGS-132 | kick-off at 6.4 s | kick-off at 6.4 s |
| SNGS-148 | goal at 24.8 s | see CLAUDE.md |

Player positions are within 0.6-1.2 m of ground truth on median.

## What does not work yet

- **The ball is often missed.** It is found in 20-78% of frames depending on the clip. Set
  pieces are robust to that; shots, goals and passes from raw video are not yet reliable.
- **Throw-ins are not detected.**
- **"Keeper throw" is a guess** (it cannot tell a throw from a short kick).
- **It is not live yet.** It processes a clip, then speaks. Rules need at most 5 s of
  look-ahead, so a version that runs a few seconds behind live play is the next step.
- **The wording is templates**, so it repeats itself.

`CLAUDE.md` has the full list, the measurements behind every model choice, and the roadmap.

## Models and licenses

| Model | Job | License |
|---|---|---|
| [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) (YOLO11n) | players, referee, ball | none stated |
| [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) | pitch landmarks | none stated |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [fastrtc/kokoro-onnx](https://huggingface.co/fastrtc/kokoro-onnx) | voice | Apache-2.0 / MIT |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | inference, tracking | AGPL-3.0 |

Two of the models state no license, so treat this as a study project until that is settled.
Test data: [SoccerNet/SN-GSR-2024](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024)
(GPL-3.0), fetched on demand and never stored in this repository.

## Development

```bash
python -m pytest -q                                   # 38 tests, under a second
python tools/fetch_gsr.py labels test data/gsr/test   # ground truth for scoring
python tools/eval_gsr.py data/gsr/test --cache data/gsr/test.pkl
```
