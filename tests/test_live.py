"""The live engine, driven frame by frame through scripted scenes. Nothing may use the future."""

import numpy as np

from commentator.commentary import Moment, TemplateNarrator
from commentator.live import Analyst, LiveCommentator, LiveDirector, latency_report
from commentator.sim import Scene, keeper, squad
from commentator.speech import SAMPLE_RATE, Booth

NAMES = {0: "White", 1: "Red"}
RIGHT_KEEPER = keeper(1, 91, (104.4, 34.0))


class FakeVoice:
    """Speaks at three words a second and costs nothing to run."""

    def synthesize(self, text, voice=None, tone="calm"):
        return np.zeros(int(SAMPLE_RATE * max(0.6, len(text.split()) / 3.0)), np.float32)


def go_live(scene: Scene, delay: float = 4.0, analyst: bool = True) -> LiveDirector:
    frames = scene.frames()
    director = LiveDirector(TemplateNarrator(NAMES), Booth(FakeVoice()),
                            TemplateNarrator(NAMES, seed=3), Analyst() if analyst else None,
                            delay_s=delay)
    live = LiveCommentator(director)
    now = 0.0
    for k, frame in enumerate(frames):
        now = frame.t
        if live.due_for_rules(now):
            live.update(frames[:k + 1], scene.fps, now)  # only what has happened so far
        live.speak(now)
    end = now
    while now < end + delay + 5.0:
        now += 1.0 / scene.fps
        live.speak(now)
    return director


def kinds(director):
    return [s.kind for s in director.aired]


def corner_scene() -> Scene:
    players = {**squad(0, 1, [(103.5, 1.5), (95, 30), (97, 36), (96, 33)]),
               **squad(1, 11, [(99, 33), (101, 37)]), **RIGHT_KEEPER}
    return Scene(players, ball=(104.5, 0.8)).hold(7.0).fly(1.2, (97.0, 35.0)).hold(4.0)


def test_corner_is_announced_while_it_waits_then_called_when_taken():
    director = go_live(corner_scene())
    said = kinds(director)
    assert said.index("corner_kick.setup") < said.index("corner_kick")
    setup = next(s for s in director.aired if s.kind == "corner_kick.setup")
    taken = next(s for s in director.aired if s.kind == "corner_kick")
    assert setup.decided_t < 7.0          # known while the ball was still waiting
    assert taken.late_by <= 0.6           # and the delivery is called as it appears on screen
    assert abs(taken.event_t - 7.0) < 0.3


def test_nothing_is_said_before_it_is_known():
    director = go_live(corner_scene())
    for line in director.aired:
        assert line.start >= line.decided_t - 1e-6
        assert line.start >= line.due - 1e-6


def test_the_analyst_follows_the_lead_and_never_talks_over_them():
    director = go_live(corner_scene())
    aired = director.aired
    assert any(s.speaker == "analyst" for s in aired)
    for a, b in zip(aired, aired[1:]):
        assert b.start >= a.end - 1e-6
    first_analyst = next(i for i, s in enumerate(aired) if s.speaker == "analyst")
    assert aired[first_analyst - 1].speaker == "lead"


def test_a_goal_call_is_taken_back_on_air_when_the_keeper_turns_up_with_the_ball():
    players = {**squad(0, 1, [(92, 36)]), **squad(1, 11, [(96, 30)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(92.0, 36.5)).hold(1.0)
             .fly(0.6, (107.0, 34.5), error=(3.0, -8.0)).hold(3.0))
    scene.put(91, (106.5, 34.3)).roll(1.5, (100.0, 34.2), carrier=91).hold(4.0)
    said = kinds(go_live(scene, analyst=False))
    assert "goal" in said and "goal.cancelled" in said
    assert said.index("goal") < said.index("goal.cancelled")


def test_without_a_delay_buffer_reactions_trail_the_action():
    director = go_live(corner_scene(), delay=0.0)
    taken = next(s for s in director.aired if s.kind == "corner_kick")
    assert taken.late_by > 0.3            # it could not be known the instant it happened
    assert latency_report(director.aired, 0.0)["lines"] == len(director.aired)


def test_a_goal_interrupts_small_talk_but_not_another_big_call():
    director = LiveDirector(TemplateNarrator(NAMES), Booth(FakeVoice()), delay_s=0.0)
    director.offer(Moment(0.0, "spell", 20, team=0), now=0.0)
    assert [s.kind for s in director.poll(0.0)] == ["spell"]
    director.offer(Moment(0.5, "goal", 100, team=1), now=0.5)
    started = director.poll(0.5)
    assert [s.kind for s in started] == ["goal"]
    assert director.aired[0].cut_at == 0.5          # the remark was cut short
    director.offer(Moment(0.8, "penalty", 90, team=0), now=0.8)
    assert director.poll(0.8) == []                 # a goal call is not talked over


def test_a_remark_that_missed_its_moment_is_dropped():
    director = LiveDirector(TemplateNarrator(NAMES), Booth(FakeVoice()), delay_s=0.0)
    director.offer(Moment(0.0, "goal", 100, team=0), now=0.0)
    director.poll(0.0)
    director.offer(Moment(0.2, "turnover", 25, team=1), now=0.2)
    for now in np.arange(0.2, 6.0, 0.1):
        director.poll(float(now))
    assert [s.kind for s in director.aired] == ["goal"]
    assert director.dropped and director.dropped[0][0] == "turnover"


def test_team_names_learned_after_the_start_are_used():
    """Live, the teams are only known once enough shirts have been seen."""
    names = {}
    narrator = TemplateNarrator(names)
    assert narrator.say(Moment(1.0, "turnover", 25, team=0)) is None
    names[0] = "White"
    assert "White" in narrator.say(Moment(1.0, "turnover", 25, team=0))


def test_silence_is_filled_when_the_booth_is_set_to_talkative():
    quiet = LiveDirector(TemplateNarrator(NAMES), Booth(FakeVoice()), delay_s=0.0, density="low")
    chatty = LiveDirector(TemplateNarrator(NAMES), Booth(FakeVoice()), delay_s=0.0,
                          analyst=Analyst(), density="high")
    for director in (quiet, chatty):
        for now in np.arange(0.0, 20.0, 0.08):
            director.poll(float(now))
    assert quiet.aired == []
    assert len(chatty.aired) >= 4
    assert {s.speaker for s in chatty.aired} == {"lead", "analyst"}   # they take turns
    for a, b in zip(chatty.aired, chatty.aired[1:]):
        assert b.start >= a.end


def test_nothing_is_spoken_before_the_delayed_picture_starts():
    director = go_live(corner_scene(), delay=4.0)
    assert all(line.start >= 4.0 for line in director.aired)
