# AnkaraMessi: an AI football commentator

Give it footage of a real match. It returns the same footage with a spoken two-voice
commentary and captions: a lead commentator calling the play and an analyst reacting to it.

It works as a live system running a few seconds behind the action, and on finished clips.
No paid API is used anywhere.

## What it does

1. Finds the players, referee and ball in each frame, and tracks the players.
2. Works out where the camera is looking, so every player gets a position on the pitch in metres.
3. Splits players into two teams by shirt colour and spots the goalkeepers.
4. Looks at a sliding window of the last 45 seconds, twice a second, and recognises what
   happened: corner, penalty, kick-off, goal kick, free kick, throw-in, shot, save, goal, and a
   stream of smaller things (who is on the ball, a run, a break, play entering the box, a team
   sitting deep).
5. Decides who speaks now. Big calls go first and can interrupt small talk; silences are filled.
6. Speaks each line in a tone that fits (excited, tense, calm, deflated) and puts it on the video.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements.txt
./tools/fetch_models.sh                      # about 450 MB into ./models

python -m commentator run my_clip.mp4 --out out/my_clip
```

Outputs in `out/my_clip/`: `commentated_live.mp4`, `commentary_live.txt` (each line with when
it happened, when the system knew, and whether it was on time), `commentary_live.wav`,
`events.json`, `live_report.json`.

Options:

| Flag | What it does |
|---|---|
| `--teams "Ankara,Istanbul"` | use real team names instead of shirt colours |
| `--density low\|normal\|high` | how talkative it is (default `high`) |
| `--overlay none\|map\|full` | `none` is a clean picture (default); `map` adds a mini-map; `full` also marks every player |
| `--llm PATH` or `--llm ollama:NAME` | let a local language model word the analyst's lines |
| `--device cuda:0` | run the vision models on a GPU |

Use wide broadcast footage from the main camera. Close-ups are skipped.

## Running it live

```bash
python -m commentator bench my_clip.mp4 --device cuda:0      # can this machine keep up?
python -m commentator live  my_clip.mp4 --out out/live --pace real --device cuda:0
```

`bench` times each stage and tells you whether the machine can analyse 12.5 frames a second.
`live --pace real` feeds the file at real speed and reports how far it fell behind.

The picture is shown 4 seconds behind the camera (`--delay`), the way TV is. That is what lets
a call land exactly on its moment: every decision is made inside those 4 seconds.

**What is known about speed:**

| Stage | 4 CPU cores | Needed for live |
|---|---|---|
| vision (detection, calibration, ball probes) | 365 ms per frame | 80 ms per frame |
| event rules | 8 ms, twice a second | fine |
| speech | 3.3x faster than real time, cached after first use | fine |
| everything after vision, at real pace | fell behind by 0.06 s | fine |

So a GPU is needed for the vision models only. Both are small networks, so a modest GPU should
be enough, but **this has not been measured on a GPU yet**. Run `bench` first.

## Results

Measured on SoccerNet GSR 2024: real 30-second broadcast clips with ground truth.

**Important moments, given correct positions.** 29 of 33 labelled actions found on the
development clips and 35 of 40 on a second set, both offline and when run live with the
picture 4 seconds behind. Before any tuning on the second set the score was 30 of 40, which is
the fair estimate for new footage. Live, a goal is known about 0.8 seconds after it happens and
a set piece is announced while the ball is still waiting.

**From raw video, four clips.**

| What really happened | What the commentary did |
|---|---|
| corner at 6.3 s | announced the corner at 1.0 s, called the delivery at 6.4 s |
| penalty at 6.6 s | announced the penalty at 1.2 s, called the kick at 6.3 s |
| kick-off at 6.4 s | announced at 0.8 s, called at 6.4 s |
| goal at 24.8 s | called a shot at 23.8 s; did **not** confirm the goal |

Each clip gets 11-12 spoken lines in 30 seconds.

## What does not work yet

- **Real-time speed is unproven.** See above: no GPU measurement yet.
- **The ball is often missed** (found in 20-78% of frames). Set pieces cope with that; goals,
  shots and passes from raw video do not yet. The test goal is reported as a shot.
- **The chatty lines are sometimes wrong.** "Into the box", "break", "sitting deep" and the
  rest are there to keep the commentary alive and have not been checked against ground truth.
- **Filler lines repeat** on longer clips and say nothing specific.
- **It has only been fed files**, not a camera or a network stream.
- **A language model for the analyst is optional and modest.** A 1.7B model rewords lines
  safely but slowly on a CPU; anything it invents is rejected and the template is used.

`CLAUDE.md` has the full list, the measurements behind every choice, and the roadmap.

## Models and licenses

| Model | Job | License |
|---|---|---|
| [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) (YOLO11n) | players, referee, ball | none stated |
| [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) | pitch landmarks | none stated |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [fastrtc/kokoro-onnx](https://huggingface.co/fastrtc/kokoro-onnx) | both voices | Apache-2.0 / MIT |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (optional) | analyst wording | Apache-2.0 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | inference, tracking | AGPL-3.0 |

Two of the models state no license, so treat this as a study project until that is settled.
Test data: [SoccerNet/SN-GSR-2024](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024)
(GPL-3.0), fetched on demand and never stored in this repository.

## Development

```bash
python -m pytest -q                                       # 64 tests, about a second
python tools/fetch_gsr.py labels test data/gsr/test       # ground truth for scoring
python tools/eval_gsr.py data/gsr/test --cache data/gsr/test.pkl    # offline score
python tools/eval_live.py data/gsr/test.pkl --delay 4               # live score and latency
```
