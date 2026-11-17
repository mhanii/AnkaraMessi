"""Every threshold the event rules use. Change them here, not inline, and say why in the commit.

Values were chosen by looking at SoccerNet GSR ground truth (real broadcast clips), not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleConfig:
    # --- Possession ---
    possession_radius_m: float = 2.5
    possession_min_s: float = 0.24
    possession_merge_gap_s: float = 0.4

    # --- Stationary ball (set pieces) ---
    still_radius_m: float = 1.0
    still_min_s: float = 1.0
    still_merge_gap_s: float = 1.0
    still_merge_dist_m: float = 1.5
    taken_move_m: float = 3.0  # ball this far from the spot = the set piece was taken
    taken_window_s: float = 5.0
    taken_confirm_s: float = 1.0  # the ball must stay away this long (one stray detection
    #                               elsewhere is common with a weak detector)
    taken_blind_s: float = 1.0  # longer than this unseen = the kick happened in the gap
    taken_refound_lag_s: float = 0.3  # a re-found ball left the spot about this long ago

    corner_radius_m: float = 3.0
    corner_min_s: float = 1.0
    penalty_radius_m: float = 2.0
    penalty_min_s: float = 2.0
    penalty_area_inset_m: float = 1.0  # players lined up on the edge do not count as inside
    penalty_max_in_area: int = 3  # taker + keeper + one positional error
    penalty_keeper_depth_m: float = 2.5  # somebody must be standing on the goal line
    kickoff_radius_m: float = 3.0  # loose: the formation check is what makes it a kick-off
    kickoff_min_s: float = 1.5
    kickoff_formation_min_s: float = 2.0
    kickoff_min_players: int = 6  # per team, to trust a formation with no ball in view
    kickoff_own_half_fraction: float = 0.85
    kickoff_halfway_slack_m: float = 1.5
    goal_kick_depth_m: tuple = (3.0, 8.0)  # ball placed on/near the 5.5 m line
    goal_kick_halfwidth_m: float = 12.0
    goal_kick_min_s: float = 2.0
    free_kick_min_s: float = 3.0
    keeper_hold_radius_m: float = 1.0  # real goal kicks: keeper this close <30% of the wait
    repeat_gap_s: float = 8.0  # the same dead ball being re-placed is one set piece
    repeat_dist_m: float = 8.0
    after_set_piece_s: float = 1.5  # the first ball out of a set piece is the set piece

    # --- Shots ---
    shot_max_range_m: float = 35.0
    shot_lookahead_s: float = 3.5
    shot_cross_within_s: float = 2.5  # ball must be past the goal line this soon after release
    shot_cross_halfwidth_m: float = 12.0  # ...and then be seen this close to the goal centre
    shot_min_speed_mps: float = 10.0
    shot_contest_radius_m: float = 3.0
    goal_net_depth_m: tuple = (-4.5, -0.2)
    goal_net_halfwidth_m: float = 3.9  # post (3.66) plus a ball's width of calibration slack
    goal_net_min_s: float = 0.7
    goal_in_play_within_s: float = 4.0  # ball must have been on the pitch this recently
    goal_link_s: float = 5.0  # a shot this soon before the ball is in the net scored it
    goal_repeat_s: float = 15.0
    reconcile_window_s: float = 30.0  # how far ahead a goal kick / corner settles a shot
    keeper_restart_window_s: float = 15.0

    # --- Passing ---
    pass_max_gap_s: float = 4.0
    pass_min_m: float = 5.0
    long_pass_m: float = 30.0
    turnover_min_hold_s: float = 1.0
    keeper_hold_min_s: float = 1.0
    hand_throw_range_m: tuple = (10.0, 40.0)
    hand_throw_max_speed_mps: float = 16.0


    # --- Throw-ins ---
    throw_band_m: float = 1.5  # thrower's feet this close to the touchline (or outside it)
    throw_outside_m: float = 4.0
    throw_min_hold_s: float = 0.5
    throw_max_drift_m: float = 2.5  # the thrower stands still; a winger running does not
    throw_merge_gap_s: float = 4.0  # the detector loses a held ball for seconds at a time
    throw_min_lift_m: float = 0.8  # a held ball projects beyond the feet; one at the feet does not
    long_throw_m: float = 18.0
    throw_receive_s: float = 4.0

    # --- Running play-by-play (chatty on purpose, not scored) ---
    play_zone_hold_s: float = 0.8  # the action must stay in a new zone this long to be called
    play_box_repeat_s: float = 6.0
    play_on_ball_min_s: float = 0.6
    play_on_ball_repeat_s: float = 2.5
    play_carry_min_s: float = 1.2
    play_carry_min_m: float = 8.0
    shape_min_s: float = 2.0
    shape_deep_zone_m: float = 35.0  # own third
    shape_deep_players: int = 8
    shape_high_line_m: float = 35.0  # deepest outfield player this far from their own goal
    shape_repeat_s: float = 15.0
    break_window_s: float = 4.0
    break_min_m: float = 28.0  # the action moved this far up the pitch within the window
    break_repeat_s: float = 8.0
    situation_every_s: float = 3.0
    numbers_radius_m: float = 12.0
    numbers_repeat_s: float = 10.0

    # --- Phases of play (no ball needed) ---
    pressure_zone_m: float = 35.0  # the attacking third
    pressure_min_players: int = 5
    pressure_min_s: float = 3.0
    pressure_repeat_s: float = 20.0
    replay_min_s: float = 3.0


DEFAULT = RuleConfig()
