"""Every event rule against a scripted scene with a known answer.

Team 0 defends the left goal (x=0), team 1 the right goal (x=105).
"""

from commentator.events import detect_events
from commentator.sim import Scene, keeper, squad
from commentator.timeline import Timeline

LEFT_KEEPER = keeper(0, 90, (1.0, 34.0))
RIGHT_KEEPER = keeper(1, 91, (104.4, 34.0))


def events_of(scene: Scene):
    return detect_events(Timeline(scene.frames(), scene.fps))


def kinds(events):
    return [e.type for e in events]


def one(events, kind):
    found = [e for e in events if e.type == kind]
    assert len(found) == 1, f"expected exactly one {kind}, got {kinds(events)}"
    return found[0]


def test_corner_kick_is_taken_when_the_ball_leaves_the_flag():
    players = {**squad(0, 1, [(103.5, 1.5), (95, 30), (97, 36)]),
               **squad(1, 11, [(99, 33), (101, 37)]), **RIGHT_KEEPER}
    scene = Scene(players, ball=(104.5, 0.8)).hold(4.0).fly(1.2, (97.0, 35.0)).hold(1.0)
    e = one(events_of(scene), "corner_kick")
    assert e.team == 0 and e.detail["corner"] == "far-right" and e.detail["taken"]
    assert abs(e.t - 4.0) < 0.3


def test_penalty_then_goal():
    # Everyone but the taker and the keeper stands on the edge of the area.
    players = {**squad(0, 1, [(91.0, 33.5), (88.3, 20), (88.2, 44)]),
               **squad(1, 11, [(88.4, 30), (88.3, 38)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(94.0, 34.0)).hold(4.0)
             .fly(0.5, (106.5, 33.0), error=(4.0, -6.0)).hold(2.0))
    events = events_of(scene)
    pen = one(events, "penalty")
    goal = one(events, "goal")
    assert pen.team == 0 and goal.team == 0
    assert pen.detail["outcome"] == "goal"
    assert "free_kick" not in kinds(events)


def test_dead_ball_on_the_spot_with_a_crowded_box_is_not_a_penalty():
    players = {**squad(0, 1, [(91.0, 33.5), (95, 30), (97, 38), (99, 33)]),
               **squad(1, 11, [(96, 31), (98, 36), (100, 34)]), **RIGHT_KEEPER}
    scene = Scene(players, ball=(94.0, 34.0)).hold(4.0).roll(1.0, (85.0, 40.0))
    assert "penalty" not in kinds(events_of(scene))


def test_kick_off_needs_both_teams_in_their_own_halves():
    left = [(51.5, 34), (40, 20), (40, 48), (30, 34), (25, 15), (25, 55)]
    right = [(60, 30), (62, 40), (70, 20), (70, 48), (80, 34), (78, 10)]
    players = {**squad(0, 1, left), **squad(1, 11, right)}
    scene = Scene(players, ball=(52.5, 34.0)).hold(3.0).roll(0.8, (44.0, 30.0)).hold(1.0)
    e = one(events_of(scene), "kick_off")
    assert abs(e.t - 3.0) < 0.3

    mixed = {**squad(0, 1, left[:4] + [(70, 30), (75, 40)]), **squad(1, 11, right)}
    scene = Scene(mixed, ball=(52.5, 34.0)).hold(3.0).roll(0.8, (44.0, 30.0)).hold(1.0)
    assert "kick_off" not in kinds(events_of(scene))


def test_kick_off_is_recognised_from_the_formation_when_the_ball_is_not_seen():
    left = [(51.8, 34), (40, 20), (40, 48), (30, 34), (25, 15), (25, 55)]
    right = [(60, 30), (62, 40), (70, 20), (70, 48), (80, 34), (78, 10)]
    scene = Scene({**squad(0, 1, left), **squad(1, 11, right)}, ball=(52.5, 34.0)).hide(3.0)
    scene.put(2, (58.0, 20.0)).put(3, (60.0, 48.0)).put(12, (45.0, 40.0)).hide(2.0)
    e = one(events_of(scene), "kick_off")
    assert e.detail["by"] == "formation" and abs(e.t - 3.0) < 0.5


def test_goal_kick():
    players = {**squad(0, 1, [(30, 30), (35, 40)]), **squad(1, 11, [(40, 34)]), **LEFT_KEEPER}
    scene = Scene(players, ball=(5.5, 38.0)).hold(3.0).fly(2.0, (45.0, 30.0)).hold(0.5)
    events = events_of(scene)
    one(events, "goal_kick")
    assert "shot" not in kinds(events) and "free_kick" not in kinds(events)


def test_free_kick():
    players = {**squad(0, 1, [(61, 21), (70, 40)]), **squad(1, 11, [(70, 22), (72, 26)])}
    scene = Scene(players, ball=(60.0, 20.0)).hold(4.0).roll(0.8, (70.0, 39.0)).hold(1.0)
    e = one(events_of(scene), "free_kick")
    assert e.team == 0


def test_short_and_long_pass_survive_an_airborne_ball():
    players = {**squad(0, 1, [(30, 30), (42, 32), (78, 50)]), **squad(1, 11, [(60, 10)])}
    scene = (Scene(players, ball=(30.0, 30.5)).hold(1.0).roll(0.8, (42.0, 32.5)).hold(1.0)
             .fly(1.8, (78.0, 50.5)).hold(1.0))
    events = events_of(scene)
    short, long = one(events, "pass"), one(events, "long_pass")
    assert short.player == 1 and short.receiver == 2 and 10 < short.distance_m < 14
    assert long.player == 2 and long.receiver == 3 and long.distance_m > 30
    assert "shot" not in kinds(events)


def test_a_dribble_is_not_a_pass():
    players = {**squad(0, 1, [(30, 30)]), **squad(1, 11, [(60, 10)])}
    scene = Scene(players, ball=(30.5, 30.0)).hold(0.5).roll(4.0, (55.0, 36.0), carrier=1)
    assert events_of(scene) == []


def test_turnover_is_called_once_the_other_team_keeps_the_ball():
    players = {**squad(0, 1, [(30, 30)]), **squad(1, 11, [(42, 32)])}
    scene = Scene(players, ball=(30.0, 30.5)).hold(1.0).roll(0.8, (42.0, 32.5)).hold(2.0)
    e = one(events_of(scene), "turnover")
    assert e.team == 1 and e.detail["from_team"] == 0


def test_shot_wide_is_settled_by_the_goal_kick_that_follows():
    players = {**squad(0, 1, [(88, 30)]), **squad(1, 11, [(95, 36)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(88.0, 30.5)).hold(1.0)
             .fly(0.7, (109.0, 41.0), error=(6.0, -14.0)).hide(3.0)
             .hold(3.0, at=(99.5, 37.0)).fly(2.0, (50.0, 30.0)))
    events = events_of(scene)
    shot = one(events, "shot")
    assert shot.team == 0 and shot.detail["outcome"] == "off_target"
    one(events, "goal_kick")
    assert "goal" not in kinds(events)


def test_goal_from_open_play_needs_the_ball_resting_in_the_net():
    players = {**squad(0, 1, [(92, 36)]), **squad(1, 11, [(96, 30)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(92.0, 36.5)).hold(1.0)
             .fly(0.6, (107.0, 34.5), error=(3.0, -8.0)).hold(2.5))
    events = events_of(scene)
    goal = one(events, "goal")
    assert goal.team == 0 and "shot" not in kinds(events)


def test_goal_without_a_seen_shot_is_still_a_goal():
    players = {**squad(0, 1, [(80, 50)]), **squad(1, 11, [(85, 20)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(97.0, 40.0)).roll(0.4, (100.0, 37.0)).hide(0.6)
             .hold(2.0, at=(106.8, 34.8)))
    goal = one(events_of(scene), "goal")
    assert goal.team is None and goal.detail["source"] == "ball_in_net"


def test_clearance_by_the_defending_team_is_not_a_shot():
    players = {**squad(1, 11, [(96, 30)]), **squad(0, 1, [(70, 30)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(96.0, 30.5)).hold(1.0)
             .fly(0.8, (110.0, 38.0), error=(4.0, -6.0)).hide(2.0))
    assert "shot" not in kinds(events_of(scene))


def test_keeper_holding_the_ball_is_not_a_goal_kick():
    players = {**squad(0, 1, [(26, 20)]), **squad(1, 11, [(50, 34)]), **keeper(0, 90, (6.0, 34.0))}
    scene = Scene(players, ball=(6.0, 34.5)).hold(4.0).roll(1.0, (12.0, 30.0), carrier=90)
    assert "goal_kick" not in kinds(events_of(scene))


def test_goal_is_withdrawn_when_the_keeper_is_then_seen_with_the_ball():
    players = {**squad(0, 1, [(92, 36)]), **squad(1, 11, [(96, 30)]), **RIGHT_KEEPER}
    scene = (Scene(players, ball=(92.0, 36.5)).hold(1.0)
             .fly(0.6, (107.0, 34.5), error=(3.0, -8.0)).hold(1.0))
    scene.put(91, (106.5, 34.3)).roll(1.5, (100.0, 34.2), carrier=91).hold(2.0)
    events = events_of(scene)
    assert "goal" not in kinds(events)
    assert one(events, "shot").detail["outcome"] == "saved"


def test_keeper_throw():
    players = {**squad(0, 1, [(26, 20)]), **squad(1, 11, [(50, 34)]), **keeper(0, 90, (6.0, 34.0))}
    scene = (Scene(players, ball=(6.0, 34.5)).hold(2.0)
             .fly(2.4, (26.0, 20.5), error=(2.0, -5.0)).hold(1.0))
    e = one(events_of(scene), "hand_throw")
    assert e.team == 0 and 18 < e.distance_m < 28


def test_a_stationary_ball_survives_long_detector_dropouts():
    """Real detectors see a waiting ball only now and then. Sightings that agree are one wait."""
    players = {**squad(0, 1, [(103.5, 1.5)]), **squad(1, 11, [(99, 33)]), **RIGHT_KEEPER}
    scene = Scene(players, ball=(104.5, 0.8)).hold(0.3)
    for _ in range(3):
        scene.hide(1.5).hold(0.2)
    scene.fly(1.2, (97.0, 35.0)).hold(1.0)
    one(events_of(scene), "corner_kick")


def test_pressure_is_called_when_a_team_camps_in_the_final_third():
    attackers = [(80, 20), (84, 30), (88, 40), (90, 25), (95, 34), (60, 34)]
    defenders = [(92, 22), (94, 30), (96, 38), (98, 28), (99, 40), (85, 34)]
    players = {**squad(0, 1, attackers), **squad(1, 11, defenders), **RIGHT_KEEPER}
    scene = Scene(players, ball=(84.0, 30.5)).hold(5.0)
    e = one(events_of(scene), "pressure")
    assert e.team == 0 and e.detail["players_forward"] == 5


def test_replay_is_a_stretch_with_no_pitch_view():
    players = {**squad(0, 1, [(30, 30)]), **squad(1, 11, [(60, 10)])}
    scene = Scene(players, ball=(30.0, 30.5)).hold(2.0)
    frames = scene.frames()
    for k in range(50, 50 + 100):
        frames.append(type(frames[0])(k, k / 25.0, None, [], calibrated=False))
    frames.append(type(frames[0])(150, 6.0, (30.0, 30.5), frames[0].players))
    e = one(detect_events(Timeline(frames, 25.0)), "replay")
    assert abs(e.t - 2.0) < 0.1 and abs(e.end_t - 5.96) < 0.1


def test_throw_in_short_and_long():
    players = {**squad(0, 1, [(40.0, 0.2), (46.0, 8.0), (62.0, 16.0)]), **squad(1, 11, [(50, 30)])}
    short = (Scene(players, ball=(40.0, 0.2)).hide(1.0).in_hands(1.5, 1)
             .fly(0.8, (46.0, 8.5), error=(0.0, -4.0)).hold(1.0))
    e = one(events_of(short), "throw_in")
    assert e.team == 0 and e.player == 1 and e.receiver == 2
    assert e.detail["touchline"] == "far" and e.distance_m < 12

    long = (Scene(players, ball=(40.0, 0.2)).hide(1.0).in_hands(1.5, 1)
            .fly(1.6, (62.0, 16.5), error=(0.0, -8.0)).hold(1.0))
    e = one(events_of(long), "long_throw")
    assert e.receiver == 3 and e.distance_m > 18
    assert "pass" not in kinds(events_of(long)) and "long_pass" not in kinds(events_of(long))


def test_a_winger_running_down_the_line_with_the_ball_is_not_a_throw_in():
    players = {**squad(0, 1, [(40.0, 0.5)]), **squad(1, 11, [(50, 30)])}
    scene = Scene(players, ball=(40.3, 0.6)).hold(0.3).roll(3.0, (62.0, 1.0), carrier=1)
    assert events_of(scene) == []


def test_a_ball_carried_while_walking_is_not_a_throw_in():
    players = {**squad(0, 1, [(40.0, 0.2)]), **squad(1, 11, [(50, 30)])}
    scene = Scene(players, ball=(40.0, 0.2)).in_hands(1.0, 1)
    scene.put(1, (46.0, 0.2)).in_hands(1.0, 1)
    assert "throw_in" not in kinds(events_of(scene))
