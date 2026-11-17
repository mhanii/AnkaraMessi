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

The human owner is a student. Everything must run on a laptop CPU or a rented/free GPU, with
no paid APIs. They want the commentary **talkative**: plenty of lines, FIFA-game style, and
they accept that the chatty lines are sometimes wrong. They want the picture **clean**: no
markers on players unless asked for.

## Mission

A live AI football commentator. Input: real match footage (tested on 30-60 second clips).
Output: the same video with a two-voice spoken commentary and captions, produced a few seconds
behind the play.

Fouls, cards and offside are out of scope for now.

## Pipeline

```
frame ─► people + ball detection (YOLO11n)            perception/detector.py
      ─► people tracking (ByteTrack, people only)      perception/detector.py
      ─► per-frame pitch calibration (29 landmarks)    perception/calibrate.py
      ─► zoom probes for the ball at set-piece spots   perception/detector.py (probe_ball)
      ─► Perception.step / snapshot (teams, keepers)   perception/pipeline.py
      ─► game state for the last 45 s, in pitch metres timeline.py
      ─► event rules over that window, every 0.5 s     events/
      ─► EventTracker: what is new since last time     live.py
      ─► lead narrator + Analyst (second voice)        commentary.py, live.py
      ─► LiveDirector: who speaks now                  live.py
      ─► Booth: tone-shaped speech, cached per line    speech.py
      ─► clean video with captions and audio           render.py
```

One engine serves both uses. `run` analyses a finished clip, then feeds the saved observations
through the live engine. `live` does it on a feed as it arrives.

```bash
python -m commentator run CLIP --out out/clip                 # a finished clip
python -m commentator live CLIP --out out/clip --pace real --device cuda:0
python -m commentator bench CLIP --device cuda:0              # can this machine run it live?
```

Useful flags: `--density low|normal|high` (default high), `--overlay none|map|full` (default
none), `--teams "A,B"`, `--delay 4`, `--llm PATH|ollama:NAME`, `--replay analysis.pkl`.

## The one fact that shapes the rules

**A ball in the air does not project to where it is.** Mapping an image position onto the flat
pitch is only correct for things on the ground. Even in hand-annotated SoccerNet ground truth,
the projected ball "travels" at hundreds of m/s and spends seconds "outside" the pitch.

So no rule may trust ball speed, line crossings, or mid-flight ball positions. Rules are built
on what survives: player positions, possession (ball at a player's feet), a ball lying still,
and where the ball ends up.

## How live works

The picture is shown `--delay` seconds behind the camera (default 4), as TV broadcasts are.
Every rule reaches its verdict inside that delay, so a line starts when its moment appears on
screen. Three consequences worth knowing:

- **The director can see what is coming.** A bigger line that is about to come due is already
  queued, so a small remark that would run into it is held back, and silence is filled only
  with a line short enough for the room available.
- **Time-critical calls interrupt small talk.** Priority 55 and above (see `PRIORITY` in
  `commentary.py`) may cut across anything below 50.
- **A live goal call cannot wait for later evidence.** Offline, a goal is withdrawn if a goal
  kick or the keeper holding the ball follows. Live, the goal is called, and if that evidence
  arrives the lead takes it back on air ("No! Hold on, it's not a goal.").

## Events

**Scored against ground truth:**

| Event | How it is recognised |
|---|---|
| corner kick | ball still within 3 m of a corner flag, then leaves |
| penalty | ball still on the spot, box empty but taker and keeper on the line |
| kick-off | ball still on the centre spot with both teams in their own halves; or that formation alone |
| goal kick | ball still near the 5.5 m line, keeper standing back from it |
| free kick | ball still elsewhere on the pitch for 3 s or more |
| shot | ball leaves an attacker within 35 m and is past the goal line near the goal within seconds |
| goal | ball comes to rest in the net zone; withdrawn if a goal kick or keeper possession follows |

**Not scored (no labels). Treat as plausible, not proven:**

| Event | How it is recognised |
|---|---|
| save, pass, long pass, turnover | possession changing hands; a keeper gathering a fast ball |
| hand throw | keeper holds the ball in the box, then a team-mate has it 10-40 m away. A guess. |
| throw-in, long throw | a still player on a touchline with the ball in the upper part of their outline. Checked by eye on one example. |
| pressure, replay | five attackers in the final third for 3 s; no pitch in view for 3 s after a big moment |

**Chatty play-by-play (`events/play.py`). Built to keep the commentary talking; expect errors:**
on the ball, carry (a run with it), pass direction, into the final third, into the box, switch
of flank, back to midfield, fast break, team sitting deep, high line, numbers around the ball,
and a plain "where the play is" every 3 s. When the ball is not detected these use the middle
of the players in view, because the broadcast camera follows the ball.

## Measured results

Dataset: SoccerNet GSR 2024 (30 s broadcast clips, per-frame pitch positions, one labelled
action per clip). `tools/fetch_gsr.py` fetches it; `tools/eval_gsr.py` and `tools/eval_live.py`
score against it.

**Event rules on ground-truth positions** (labelled actions found):

| | development split (49 clips) | second split (58 clips) |
|---|---|---|
| first score, before looking at its failures | n/a | 30 / 40 |
| offline, current | 29 / 33 | 35 / 40 |
| **live, called on air, picture 4 s behind** | **29 / 33** | **35 / 40** |

The 30/40 is the honest generalisation number. The second split has since been studied, so it
is no longer held out. Misses are almost all shots where the ball was never seen leaving a player.

**How soon the live engine knows** (median / worst, after the event, on ground truth):
goal 0.7-0.8 s / 1.1 s; shot 1.0-1.4 s / 3.6 s; set piece taken 1.4-1.7 s / 2.6 s; a set piece
being prepared is announced 0.9-2.6 s into the wait, before the kick. About 65% of lead lines
start within 0.3 s of their moment on the delayed picture; the late ones are low-priority
remarks waiting their turn. The rule engine takes 8 ms per run.

**From raw video, four real clips** (vision models, then the live engine):

| Clip | Labelled action | Called from video |
|---|---|---|
| SNGS-116 | corner at 6.3 s | corner announced at 1.0 s, delivery called at 6.4 s |
| SNGS-144 | penalty at 6.6 s | penalty announced at 1.2 s, kick called at 6.3 s |
| SNGS-132 | kick-off at 6.4 s | announced at 0.8 s, called at 6.4 s |
| SNGS-148 | goal at 24.8 s | called as a **shot** at 23.8 s; the goal is not confirmed |

11-12 lines per 30 s clip, two voices. Player positions within 0.6-1.2 m of ground truth on median.

**Perception** (200 sampled frames against ground truth): people found 94%; ball found 39% in
the whole frame (**the weakest link**), usable at set-piece spots thanks to zoom probes; pitch
located in 59-98% of frames.

**Speed, on 4 CPU cores** (`bench`): detect 99 ms + calibrate 168 ms + probes 98 ms = 365 ms
per analysed frame, i.e. 2.7 frames per second. Live at 12.5 fps needs 80 ms. **The vision
models need a GPU; this has not been measured on one.** Everything after vision keeps up on the
CPU: run at wall-clock pace from saved observations it fell behind by 0.06 s; speech is made
3.3x faster than real time and every line is cached after first use.

**Language model as the analyst** (optional, `--llm`): Qwen3-1.7B reworded 10 of 10 lines
acceptably, about 3 s per line on a busy CPU, mostly close paraphrases. In a live run it meant
fewer lines aired (11 instead of 13) because the analyst arrives late. Qwen3-0.6B copied the
prompt's example sentences, including a team name that was not in the match; that is now
rejected. Templates remain the default.

## Model decisions (all measured on the same frames)

| Job | Chosen | Rejected, and why |
|---|---|---|
| people + ball | Soccana YOLO11n | martinjolif YOLO11m (83% people, 3.5x slower); uisikdag YOLOv8 (58% people, 15% ball) |
| pitch landmarks | Soccana 29-keypoint | martinjolif 32-keypoint (landmarks off by 2-40 m on these camera angles) |
| goalkeepers | odd shirt colour + stays near goal + deepest player (no model) | martinjolif YOLO11m never labelled the keeper as one |
| ball recall | zoom probes at calibrated spots | 3-tile full-resolution inference (+4% recall for 4x cost) |
| speech | Kokoro-82M ONNX | tiny-voca: could not be installed in the build sandbox, untested |
| analyst wording | templates; Qwen3-1.7B optional | Qwen3-0.6B (parrots the prompt) |

Looked at but not usable: RF-DETR SoccerNet checkpoint (does not load with the current `rfdetr`
library, 128M parameters); two YOLO26 football models whose own cards report ball AP of 0.06.

## Known limitations

- **Not measured on a GPU.** Real-time vision is an expectation from model size, not a result.
- **Ball detection** finds the ball in 20-78% of frames depending on the clip. Shots, goals and
  passes from raw video depend on it and are unreliable. Fix: fine-tune a detector on GSR ball
  boxes.
- **The goal in the test clip is reported as a shot.** Goals from raw video are unproven.
- **The chatty lines are unverified** and some will be wrong. That was a deliberate trade.
- **Filler lines say nothing about the match** ("Tight, this.") and repeat on long clips.
- **Overlays are not causal.** The optional mini-map and markers are drawn from the final
  analysis, not from what was known at each instant. The commentary itself is causal.
- **Live mode has only been fed files**, not a camera or network stream, and speech is
  produced in the main loop rather than a worker thread.
- **Team attribution** is the nearest player's shirt colour. Checked by eye on one penalty.
- **Model licenses.** The two Soccana models have no license on their cards. Ultralytics is
  AGPL-3.0. Fine for study; sort this out before distributing anything.
- **The calibration model may have seen similar footage.** Expect worse on other leagues.

## Layout

```
commentator/
  core.py            Detection, PlayerState, FrameState, Event
  pitch.py           geometry, zones, landmark table
  config.py          every threshold, with the reason it has that value
  timeline.py        ball gap-fill, stationary windows
  possession.py      who has the ball
  events/            setpieces, shots, passing, throwins, phases, play (chatty), engine
  perception/        detector, calibrate, teams, pipeline (Perception: step / ingest / snapshot)
  commentary.py      templates, tones, priorities, template and LLM narrators, offline director
  live.py            EventTracker, Analyst, LiveDirector, LiveCommentator, latency report
  live_run.py        run_live (file or replay, virtual or real pace), bench
  speech.py          tones, Kokoro backend, Booth (line cache), mixing
  render.py          video with captions; optional overlays
  gsr.py             SoccerNet GSR ground-truth loader
  sim.py             scripted scenes for tests (with airborne-ball errors built in)
  run.py, __main__.py
tests/               64 tests; `python -m pytest -q`
tools/               fetch_models.sh, fetch_gsr.py, eval_gsr.py, eval_live.py
```

## Conventions

- A rule change is not done until `tools/eval_gsr.py` **and** `tools/eval_live.py` have been
  run on both splits and the numbers in this file are updated.
- Every rule has a scripted scene in `tests/test_rules.py`. Scenes must use `Scene.fly` for
  anything airborne, so tests cannot pass on assumptions real data breaks.
- Live behaviour is tested by feeding scenes frame by frame (`tests/test_live.py`). Nothing in
  `live.py` may look at the future.
- Thresholds live in `config.py` with a comment saying where the value came from.
- `models/`, `data/`, `out/` are git-ignored. Never commit weights, footage or dataset files.
- Facts in commentary come only from events. A narrator may choose words, never facts. A
  language model's line is rejected if it names another team, adds a number or number word,
  or introduces a capitalised name that was not in the line it was asked to reword.

## Roadmap

1. **Measure on a GPU.** `bench --device cuda:0`, then `live --pace real --device cuda:0`.
   If it does not reach 12.5 fps: calibrate every second frame, or export to TensorRT.
2. **Fine-tune a ball detector** on GSR ball boxes. Biggest accuracy gain available.
3. **Real streams.** Read RTSP/HLS/webcam; move speech and wording to worker threads; play
   audio out instead of writing a file.
4. **Score the unscored.** Label 20 throw-ins and 50 chatty lines by eye and measure.
5. **A better analyst.** A larger language model on the GPU, with match memory (score, who has
   been on top) so remarks build on each other.
6. Track the camera between calibrated frames (optical flow) to cover close-ups.

## Decision log

- **2026-11-16. Threw away the first design.** It assumed one camera calibration per clip, used
  a detector chosen without testing, and its rules read ball speed and line crossings. Real
  ground truth showed all three were wrong. The scripted tests had hidden it because the
  scripted ball never left the ground.
- **2026-11-16. Rules, not a learned action-spotter, for events.** Events are geometry and
  timing; rules are explainable, need no training, and scored 30/40 on unseen clips.
- **2026-11-16. Detect events first, word them second.** Published attempts to have a
  video-language model commentate live report it cannot follow fast action.
- **2026-11-16. Zoom where the calibration says to look.** A waiting ball is ten pixels wide.
  Knowing where the corner flags and penalty spots are lets the detector magnify only those.
- **2026-11-16. Use the projection error as a signal for throw-ins.** A ball in the hands
  projects beyond the holder's feet and sits in the upper part of their outline.
- **2026-11-16. Probing every second frame was tried and reverted.** It lost the corner.
- **2026-11-16. Time a set piece from the ball reappearing**, since the detector loses a
  waiting ball seconds before the kick.
- **2026-11-17. A delay buffer instead of chasing zero latency.** Four seconds makes every
  call land on its moment and lets the director plan around lines it knows are coming.
- **2026-11-17. One engine for clips and live.** `run` replays saved observations through the
  live engine, so there is one behaviour to test and the offline director is no longer the
  product path.
- **2026-11-17. Talkative by default, at the owner's request.** A chatty play-by-play layer and
  silence-filling were added knowing they are unverified. They are kept apart from the scored
  rules (`events/play.py`, priorities under 50) so they can never displace a real call.
- **2026-11-17. Ask the language model to reword, not to invent.** Given a list of facts, small
  models produced nonsense that still passed a facts check. Given the template line to
  rephrase, a 1.7B model stays on topic.
- **2026-11-17. Goalkeeper rule tightened** after it labelled a penalty taker as a keeper
  (odd shirt, standing in the box), which made the penalty read as a free kick.
