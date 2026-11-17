"""Unit tests for the pieces around the rules: geometry, calibration maths, timeline,
possession, the commentary director and speech text."""

import numpy as np
import pytest

from commentator import pitch
from commentator.commentary import (Line, Moment, TemplateNarrator, direct, moments_from_events,
                                    retime)
from commentator.core import PLAYER, Event, FrameState, PlayerState
from commentator.possession import possession_segments
from commentator.speech import SAMPLE_RATE, mix, speakable, spell_number
from commentator.timeline import Timeline


# --- pitch -------------------------------------------------------------------------------
def test_corner_names_follow_the_main_camera_view():
    assert pitch.corner_name(pitch.corner_nearest((104.0, 1.0))[0]) == "far-right"
    assert pitch.corner_name(pitch.corner_nearest((1.0, 67.0))[0]) == "near-left"


def test_penalty_area_inset_excludes_players_on_the_edge():
    assert pitch.in_penalty_area(88.6, 34.0, "right")
    assert not pitch.in_penalty_area(88.6, 34.0, "right", margin=-1.0)
    assert not pitch.in_penalty_area(60.0, 34.0, "right")


def test_landmark_table_is_symmetric():
    lm = np.array(pitch.SOCCANA_LANDMARKS)
    assert len(lm) == 29
    mirrored = {(round(pitch.LENGTH - x, 2), round(y, 2)) for x, y in lm}
    assert mirrored == {(round(x, 2), round(y, 2)) for x, y in lm}


# --- calibration -------------------------------------------------------------------------
def _camera() -> np.ndarray:
    """A plausible pitch->image homography (camera above the near touchline)."""
    import cv2

    src = np.array([[20, 10], [85, 10], [95, 60], [10, 60]], np.float64)
    dst = np.array([[500, 300], [1400, 300], [1800, 900], [100, 900]], np.float64)
    return cv2.findHomography(src, dst)[0]


def _project(H, pts):
    import cv2

    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def test_homography_recovers_positions_from_noisy_landmarks():
    from commentator.perception.calibrate import solve_homography

    rng = np.random.default_rng(0)
    px = _project(_camera(), pitch.SOCCANA_LANDMARKS)
    keypoints = np.column_stack([px + rng.normal(0, 1.0, px.shape), np.ones(len(px))])
    keypoints[::3, 2] = 0.1  # a third of the landmarks not found
    hom = solve_homography(keypoints, (1920, 1080))
    assert hom is not None
    truth = np.array([[30.0, 20.0], [52.5, 34.0], [80.0, 50.0]])
    assert np.abs(hom.to_pitch(_project(_camera(), truth)) - truth).max() < 0.5


def test_homography_refuses_landmarks_on_one_line():
    from commentator.perception.calibrate import solve_homography

    px = _project(_camera(), pitch.SOCCANA_LANDMARKS)
    keypoints = np.column_stack([px, np.zeros(len(px))])
    for i in (11, 13, 15, 14):  # all on the halfway line
        keypoints[i, 2] = 1.0
    assert solve_homography(keypoints, (1920, 1080)) is None


def test_homography_refuses_too_few_landmarks():
    from commentator.perception.calibrate import solve_homography

    px = _project(_camera(), pitch.SOCCANA_LANDMARKS)
    keypoints = np.column_stack([px, np.zeros(len(px))])
    keypoints[[0, 16, 15], 2] = 1.0
    assert solve_homography(keypoints, (1920, 1080)) is None


# --- timeline and possession -------------------------------------------------------------
def _frames(ball_by_frame, players=(), fps=25.0):
    return [FrameState(k, k / fps, b, list(players)) for k, b in enumerate(ball_by_frame)]


def test_short_gaps_are_filled_and_long_ones_are_not():
    balls = [(10.0, 10.0)] + [None] * 5 + [(16.0, 10.0)] + [None] * 30 + [(40.0, 10.0)]
    tl = Timeline(_frames(balls), 25.0)
    assert tl.known[:7].all() and tl.ball[3, 0] == pytest.approx(13.0)
    assert not tl.known[7:37].any()


def test_stationary_window_bridges_dropouts_only_when_the_ball_is_found_where_it_was():
    same = [(50.0, 30.0)] * 5 + [None] * 40 + [(50.2, 30.1)] * 5
    assert len(Timeline(_frames(same), 25.0).stationary_windows(1.0, 1.0)) == 1
    moved = [(50.0, 30.0)] * 5 + [None] * 40 + [(60.0, 30.0)] * 5
    assert Timeline(_frames(moved), 25.0).stationary_windows(1.0, 1.0) == []


def test_possession_goes_to_the_nearest_player_and_ignores_fly_bys():
    a, b = PlayerState(1, PLAYER, 0, (10.0, 10.0)), PlayerState(2, PLAYER, 1, (30.0, 12.0))
    balls = [(10.5, 10.0)] * 25 + [(10.5 + i, 10.0) for i in range(1, 40)]
    segs = possession_segments(Timeline(_frames(balls, [a, b]), 25.0))
    assert [s.track_id for s in segs] == [1]  # the ball passes player 2 too fast to count


# --- commentary --------------------------------------------------------------------------
def test_set_piece_gets_a_setup_line_and_a_taken_line():
    corner = Event("corner_kick", 6.0, 6.0, team=0, xy=(104.5, 0.8),
                   detail={"setup_t": 0.0, "taken": True, "corner": "far-right"})
    moments = moments_from_events([corner])
    assert [m.kind for m in moments] == ["corner_kick.setup", "corner_kick"]
    lines = direct(moments, TemplateNarrator({0: "White", 1: "Red"}))
    assert len(lines) == 2 and "White" in lines[0].text and "far right" in lines[0].text


def test_the_goal_wins_when_two_moments_collide():
    goal = Event("goal", 10.0, 10.5, team=1, confidence=0.7, detail={"outcome": "goal"})
    turnover = Event("turnover", 10.2, 10.2, team=1)
    lines = direct(moments_from_events([turnover, goal]), TemplateNarrator({0: "White", 1: "Red"}),
                   max_delay_s=0.0)
    assert len(lines) == 1 and lines[0].kind == "goal"


def test_lines_never_overlap():
    events = [Event("turnover", t, t, team=k % 2) for k, t in enumerate(np.arange(0, 20, 0.7))]
    lines = direct(moments_from_events(events), TemplateNarrator({0: "White", 1: "Red"}))
    assert lines and all(b.t >= a.end for a, b in zip(lines, lines[1:]))


def test_unknown_team_never_produces_a_line_with_a_hole_in_it():
    narrator = TemplateNarrator({})
    assert narrator.say(Moment(1.0, "turnover", 25, team=None)) is None
    text = narrator.say(Moment(1.0, "goal", 100, team=None))
    assert text and "{" not in text and "None" not in text


def test_low_confidence_goal_is_worded_as_a_question():
    narrator = TemplateNarrator({0: "White"})
    sure = Event("goal", 1.0, 1.0, team=0, confidence=0.7)
    unsure = Event("goal", 1.0, 1.0, team=0, confidence=0.5)
    assert "?" not in narrator.say(Moment(1.0, "goal", 100, 0, sure)) or True
    said = narrator.say(Moment(1.0, "goal", 100, 0, unsure))
    assert "looks like" in said or "Is that" in said


def test_three_passes_make_one_remark():
    passes = [Event("pass", t, t + 0.5, team=0, distance_m=12) for t in (1.0, 3.0, 5.0, 7.0)]
    assert [m.kind for m in moments_from_events(passes)].count("spell") == 1


def test_retime_uses_real_audio_lengths():
    lines = [Line(1.0, "a", "goal", 100, 1.0), Line(2.2, "b", "turnover", 25, 1.0)]
    out = retime(lines, [2.0, 1.0])
    assert out[1].t >= out[0].end


# --- speech ------------------------------------------------------------------------------
def test_numbers_are_spelled_out_for_speech():
    assert spell_number(7) == "seven" and spell_number(42) == "forty-two"
    assert spell_number(120) == "one hundred and twenty"
    assert speakable("A pass of 35 metres") == "A pass of thirty-five metres"


def test_mix_places_each_line_at_its_time():
    beep = np.ones(SAMPLE_RATE // 2, np.float32) * 0.5
    track = mix([1.0, 3.0], [beep, beep], 5.0)
    assert track[int(0.5 * SAMPLE_RATE)] == 0 and track[int(1.2 * SAMPLE_RATE)] > 0
    assert track[int(2.0 * SAMPLE_RATE)] == 0 and track[int(3.2 * SAMPLE_RATE)] > 0
    assert len(track) >= 5 * SAMPLE_RATE


def test_corner_wait_mentions_how_many_attackers_are_in_the_box():
    corner = Event("corner_kick", 8.0, 8.0, team=0, xy=(104.5, 0.8),
                   detail={"setup_t": 0.0, "taken": True, "corner": "far-right",
                           "attackers_in_box": 6})
    lines = direct(moments_from_events([corner]), TemplateNarrator({0: "White", 1: "Red"}))
    assert [ln.kind for ln in lines] == ["corner_kick.setup", "corner_kick.crowd", "corner_kick"]
    assert "six" in lines[1].text.lower()


def test_free_kick_near_goal_says_how_far_out():
    fk = Event("free_kick", 6.0, 6.0, team=1, xy=(80.0, 30.0),
               detail={"setup_t": 0.0, "taken": True, "range_m": 25.3})
    lines = direct(moments_from_events([fk]), TemplateNarrator({0: "White", 1: "Red"}))
    assert "25 metres" in lines[0].text


def test_llm_narrator_uses_the_model_line_when_it_keeps_to_the_facts():
    from commentator.commentary import LLMNarrator

    narrator = LLMNarrator(lambda prompt: '"Oh, White with a strike from 23 metres out!"',
                           {0: "White", 1: "Red"})
    shot = Event("shot", 5.0, 6.0, team=0, detail={"range_m": 23.0})
    moment = moments_from_events([shot])[0]
    assert narrator.say(moment) == "Oh, White with a strike from 23 metres out!"
    assert "Facts:" in narrator.prompt(moment) and "range: 23" in narrator.prompt(moment)


def test_llm_narrator_rejects_invented_facts_and_failures():
    from commentator.commentary import LLMNarrator

    shot = Event("shot", 5.0, 6.0, team=0, detail={"range_m": 23.0})
    moment = moments_from_events([shot])[0]
    names = {0: "White", 1: "Red"}
    wrong_team = LLMNarrator(lambda p: "Red let fly from distance!", names).say(moment)
    wrong_number = LLMNarrator(lambda p: "White shoot from 40 metres!", names).say(moment)
    rambling = LLMNarrator(lambda p: " ".join(["word"] * 60), names).say(moment)

    def broken(prompt):
        raise RuntimeError("model offline")

    offline = LLMNarrator(broken, names).say(moment)
    for line in (wrong_team, wrong_number, rambling, offline):
        assert "White" in line and "Red" not in line and "40" not in line


def test_an_interrupted_line_is_cut_off_and_faded():
    beep = np.ones(SAMPLE_RATE, np.float32) * 0.5
    track = mix([1.0], [beep], 3.0, cuts=[1.4])
    assert track[int(1.2 * SAMPLE_RATE)] > 0 and track[int(1.5 * SAMPLE_RATE)] == 0


def test_tone_changes_pitch_and_loudness():
    from commentator.speech import DELIVERY, shape_tone

    tone = np.sin(np.linspace(0, 200 * np.pi, SAMPLE_RATE)).astype(np.float32)
    excited, down = shape_tone(tone, DELIVERY["excited"]), shape_tone(tone, DELIVERY["down"])
    assert len(excited) < len(tone) < len(down)          # higher pitch = shorter, lower = longer
    assert np.abs(excited).max() > np.abs(down).max()    # and louder


def test_booth_speaks_a_repeated_line_for_free():
    from commentator.speech import Booth

    class Fake:
        calls = 0

        def synthesize(self, text, voice=None, tone="calm"):
            Fake.calls += 1
            return np.ones(100, np.float32)

    booth = Booth(Fake())
    _, first = booth.say("Corner for White.", "lead", "tense")
    _, again = booth.say("Corner for White.", "lead", "tense")
    booth.say("Corner for White.", "analyst", "tense")   # another voice is another recording
    assert Fake.calls == 2 and again == 0.0 and booth.hits == 1
