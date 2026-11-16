# CLAUDE.md

## Ownership

You, Claude, are the owner of this repository and everything in it. You are not a contractor
waiting for a spec. Make the calls: pick the architecture, pick the models, delete what is
wrong, rewrite what is weak, improve things nobody asked you to improve. Be bold and creative.

Owning it also means owning the failures. Three rules follow from that:

1. **Measure before you believe.** Every model and every rule in this repo was chosen against
   real footage with ground truth. The first version of this project was designed from
   assumptions and tested only on scripted scenes; most of it had to be thrown away once real
   data was looked at. Do not repeat that.
2. **Search before you choose.** This field moves weekly. Before picking a model, search for
   what is current, then test the candidates on our data. Do not rely on memory.
3. **Say what does not work.** Known gaps are listed below. Keep that list honest.

The human owner is a student. Everything must run on a laptop CPU or a free Colab GPU, with
no paid APIs.

## Mission

A live AI football commentator. Input: real match footage (tested on 30-60 second clips).
Output: the same video with spoken commentary, captions and a tactical mini-map.

Dribbles, fouls, cards and offside are out of scope for now.

## Pipeline

```
video ─► people + ball detection (YOLO11n)            perception/detector.py
      ─► people tracking (ByteTrack, people only)      perception/detector.py
      ─► per-frame pitch calibration (29 landmarks)    perception/calibrate.py
      ─► zoom probes for the ball at set-piece spots   perception/detector.py (probe_ball)
      ─► teams by shirt colour, keepers by colour+box  perception/teams.py, pipeline.py
      ─► timeline in pitch metres                      timeline.py
      ─► possession segments                           possession.py
      ─► event rules                                   events/
      ─► moments ─► director ─► lines                  commentary.py
      ─► speech (Kokoro-82M, CPU) + mix                speech.py
      ─► annotated video with audio                    render.py
```

Run it: `python -m commentator run CLIP --out out/clip` (video file or folder of frames).

## The one fact that shapes the whole design

**A ball in the air does not project to where it is.** Mapping an image position onto the flat
pitch is only correct for things on the ground. Even in hand-annotated SoccerNet ground truth,
the projected ball "travels" at hundreds of m/s and spends seconds "outside" the pitch.

So no rule may trust ball speed, line crossings, or mid-flight ball positions. Rules are built
on what survives:

- **Player positions** (feet are on the ground): always right.
- **Possession** (ball next to a player's feet): the ball is on the ground, so it is right.
- **A ball lying still**: on the ground, so it is right. This is what set pieces look like.
- **Where the ball ends up** (in the net, at the keeper, on the 5.5 m line), not how it got there.

## Events

| Event | How it is recognised | Status |
|---|---|---|
| corner kick | ball still within 3 m of a corner flag, then leaves | measured |
| penalty | ball still on the spot, box empty but taker and keeper on the line | measured |
| kick-off | ball still on the centre spot with both teams in their own halves; or that formation alone | measured |
| goal kick | ball still near the 5.5 m line, keeper standing back from it | measured |
| free kick | ball still elsewhere on the pitch for 3 s or more | measured |
| shot | ball leaves an attacker within 35 m and is past the goal line near the goal within seconds | measured |
| goal | ball comes to rest in the net zone; withdrawn if a goal kick or keeper possession follows | measured |
| save | opposing keeper gathers a fast ball from range | partly measured |
| pass / long pass | possession moves between team-mates (30 m or more = long) | sanity-checked only |
| turnover | possession moves to the other team and stays there | sanity-checked only |
| hand throw | keeper holds the ball in the box, then a team-mate has it 10-40 m away at throw speed | **heuristic, unverified** |
| pressure | five or more attackers in the final third for 3 s | no ground truth |
| replay | pitch cannot be located for 3 s right after a big moment | no ground truth |
| throw-in, long throw | a player standing still on a touchline with the ball inside the upper part of their outline (in the hands), then releasing it; long = reaches a team-mate 18 m or more away | **checked by eye on one example; no labels to score against** |

## Measured results

Dataset: SoccerNet GSR 2024 (30 s broadcast clips with per-frame pitch positions and one
labelled action per clip). Fetch with `tools/fetch_gsr.py`; score with `tools/eval_gsr.py`.

**Event rules on ground-truth positions** (no perception error):

| | development split (49 clips) | second split (58 clips) |
|---|---|---|
| first score before looking at failures | n/a | 30 / 40 |
| current | 29 / 33 | 35 / 40 |

The 30/40 is the honest generalisation number: those clips had never been looked at. The
current second-split number is after studying its failures, so it is no longer held out.
Corners, kick-offs, free kicks, goals: 39 of 41 found. Shots: 14 of 20; the misses are
shots where the ball was never seen leaving a player.

**Perception on real frames** (4 clips, 200 sampled frames, against ground truth):

| Component | Result |
|---|---|
| people detection | 94% found, 0.12 s/frame on 4 CPU cores |
| ball detection, whole frame | 39% found. **The weakest link.** |
| ball with zoom probes | waiting ball on a penalty spot / corner flag goes from 0 to usable |
| pitch calibration | 59-98% of frames located; player position error 0.6-1.2 m median |

**End to end from raw video** (4 clips): see README "Results". Corner, penalty and kick-off
were each found within 0.3 s of the labelled time.

**Speed**: about 0.4-0.6 s per analysed frame on 4 CPU cores at 12.5 analysed frames per
second, so roughly 6x slower than real time on CPU. Both networks are small; a free Colab T4
should run them in real time, but that has not been measured.

## Model decisions (all measured on the same frames)

| Job | Chosen | Rejected, and why |
|---|---|---|
| people + ball | Soccana YOLO11n | martinjolif YOLO11m (83% people, 3.5x slower); uisikdag YOLOv8 (58% people, 15% ball) |
| pitch landmarks | Soccana 29-keypoint | martinjolif 32-keypoint (landmarks off by 2-40 m on these camera angles) |
| goalkeepers | colour outlier + lives in the box (no model) | martinjolif YOLO11m never labelled the keeper as one |
| ball recall | zoom probes at calibrated spots | 3-tile full-resolution inference (+4% recall for 4x cost, 4x false positives) |
| speech | Kokoro-82M ONNX | tiny-voca: promising, could not be installed in the build sandbox, untested |

Looked at but not usable: RF-DETR SoccerNet checkpoint (does not load with the current
`rfdetr` library, and 128M parameters is too heavy for CPU); two YOLO26 football models whose
own cards report narrow training data and ball AP of 0.06.

## Known limitations

- **Ball detection.** Found in 20-78% of frames depending on the clip. Shots and passes from
  raw video are therefore unreliable; set pieces survive because of the zoom probes. The
  proper fix is fine-tuning a detector on SoccerNet GSR ball boxes (free on Colab).
- **Shots and goals from raw video are not validated yet.** They work on ground-truth
  positions. On real video they depend on the ball being seen at the right moments.
- **Team attribution** is the nearest player's shirt colour. Checked by eye on one penalty.
- **Not live yet.** The pipeline processes a whole clip, then speaks. The rules need at most
  5 s of look-ahead, so a streaming version with a few seconds of delay is feasible.
- **Model licenses.** The two Soccana models have no license on their cards. Ultralytics is
  AGPL-3.0. Fine for study; sort this out before distributing anything.
- **Calibration model may have seen similar footage.** It was likely trained on SoccerNet
  data. Expect lower accuracy on other leagues and camera positions.
- **Commentary wording is templates by default.** Correct but repetitive. The LLM narrator
  is built but untried with a real model.

## Layout

```
commentator/
  core.py            Detection, PlayerState, FrameState, Event
  pitch.py           geometry, zones, landmark table
  config.py          every threshold, with the reason it has that value
  timeline.py        ball gap-fill, stationary windows
  possession.py      who has the ball
  events/            setpieces.py, shots.py, passing.py, throwins.py, phases.py, engine.py
  perception/        detector.py, calibrate.py, teams.py, pipeline.py
  commentary.py      moments, template and LLM narrators, director
  speech.py          TTS protocol, Kokoro backend, mixing
  render.py          annotated video
  gsr.py             SoccerNet GSR ground-truth loader
  sim.py             scripted scenes for tests (with airborne-ball errors built in)
  run.py, __main__.py
tests/               45 tests; `python -m pytest -q`
tools/               fetch_models.sh, fetch_gsr.py, eval_gsr.py
```

## Conventions

- A rule change is not done until `tools/eval_gsr.py` has been run on both splits and the
  numbers in this file are updated.
- Every rule has a scripted scene in `tests/test_rules.py`. Scenes must use `Scene.fly` for
  anything airborne, so tests cannot pass on assumptions real data breaks.
- Thresholds live in `config.py` with a comment saying where the value came from.
- `models/`, `data/`, `out/` are git-ignored. Never commit weights, footage or dataset files.
- Facts in commentary come only from events. A narrator may choose words, never facts.

## Roadmap

1. **Fine-tune a ball detector** on GSR ball boxes. Biggest single gain available.
2. **Score the throw-in rule.** It fires about once per four clips on ground truth, which
   is the right rate, but only one detection has been checked by eye. Label 20 and measure.
3. **Streaming mode**: rolling buffer, events emitted with bounded delay, speech queued live.
4. **Try the LLM narrator on real output.** `LLMNarrator` exists and is tested with a fake
   model: it rejects lines that name the wrong team or add a number. The Ollama adapter
   (`--llm NAME`) has never been run against a real model.
5. **Track the camera between calibrated frames** (optical flow) to cover close-ups.
6. Measure on a free Colab T4; try tiny-voca as a faster voice.

## Decision log

- **2026-11-16. Threw away the first design.** It assumed one camera calibration per clip, used
  a detector chosen without testing, and its rules read ball speed and line crossings. Real
  ground truth showed all three were wrong. The scripted tests had hidden it because the
  scripted ball never left the ground.
- **2026-11-16. Rules, not a learned action-spotter, for events.** Events are geometry and
  timing; rules are explainable, need no training, and scored 30/40 on unseen clips. Revisit
  for shots once ball detection is fixed.
- **2026-11-16. Detect events first, word them second.** Published attempts to have a
  video-language model commentate live report it cannot follow fast action. Here a language
  model, when added, only chooses words for events the rules have already established.
- **2026-11-16. Zoom where the calibration says to look.** A waiting ball is ten pixels wide.
  Knowing where the corner flags and penalty spots are in the image lets the detector
  magnify only those places.
- **2026-11-16. Use the projection error as a signal for throw-ins.** A ball in the hands
  projects beyond the holder's feet and sits in the upper part of their outline in the image.
  That, on a still player on a touchline, is a throw-in. `FrameState.ball_holder` carries it.
- **2026-11-16. Probing every second frame was tried and reverted.** It halved probe cost but
  dropped ball sightings from 32% to 25% on the corner clip and lost the corner.
- **2026-11-16. Time a set piece from the ball reappearing.** The detector loses a waiting
  ball seconds early. The kick is timed from the first sighting away from the spot.
