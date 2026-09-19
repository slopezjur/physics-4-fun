"""Backend-independent walking parameters and immutable, per-environment command sampling."""
from dataclasses import dataclass
import math

# Command sampling ranges, in the pelvis frame. Forward is generous, lateral deliberately small -
# a sidestep is a different gait and asking for both at once teaches neither.
CMD_FORWARD = (-0.3, 1.0)      # m/s
CMD_LATERAL = (-0.25, 0.25)    # m/s
CMD_TURN = (-1.0, 1.0)         # rad/s
COMMAND_SECONDS = 5.0          # resample this often, so one episode covers several commands
# **Heading hold.** A zero yaw command used to mean "any yaw rate, just not much of one": the reward
# charged the RATE, never the accumulated heading, so the first stepping gait circled right at
# -0.24 rad/s (575 deg per 40 s) for a flat 0.95/step. Now a zero yaw command holds the heading the
# command began on - the yaw-rate command becomes a correction toward it, recomputed every step
# (legged_gym's heading command) - so drift compounds instead of costing a constant fee.
HEADING_GAIN = 0.5             # rad/s of commanded turn per rad of heading error

# **0.25, not 0.35.** 0.35 s of clearance above 6 cm is a running stride; a walking swing keeps one
# foot loaded almost throughout. The old value paid for exaggerated stepping, which is half of what
# "weird movements" means when watching this body.
TARGET_AIR_TIME = 0.25

# **Told to stand, stand.** Every walk brain on 2026-09-11 walked away from a stand command - 4 to
# 10 m in 40 s - although the tracking term charged it about 2.5 per step for the drift, and the
# reward saw the drift exactly (check: 0.114 m/s read, 0.114 real). The gait terms are gated off for
# a stand, but nothing PAID for both feet down, so stepping in place was never cheaper than walking
# on. Paid only while the command is a stand.
STAND_PLANTED = 1.5

# **Commands are sampled the way a JOYSTICK produces them**, not as three independent uniforms.
# Independent sampling spends most of the budget on combinations no player will ever send - full
# reverse with a hard turn and a sidestep - while the two that matter most are almost never drawn:
# a straight line, and turning on the spot. These are the slices, and they sum to 1.
MIX_STAND = 0.10               # zero command: a walk that cannot stop is not controllable
MIX_STRAIGHT = 0.30            # forward only, no turn
MIX_TURN_IN_PLACE = 0.20       # stand still and rotate - what a stick does when you only turn
MIX_ARC = 0.40                 # forward AND turn together, correlated: the normal case

# --- TASK STAGES ------------------------------------------------------------------------------
#
# **The reward is not staged; the TASK is.** Every term stays active at every stage - a gait that is
# only balanced at stage 3 was never a gait, and turning a term on later cannot repair a policy that
# converged without it. What changes is how much is asked at once, because the measured failure was
# EXPLORATION: seeded from a brain that had spent hours learning to stand perfectly still, the
# policy could not discover a step, and asking it to also turn, sidestep and reverse in the same
# session spreads that search thinner still.
#
#   1  walk FORWARD, 0.15-0.35 m/s. One thing to discover.
#   2  the full speed range including reverse and sidestep, still no turning.
#   3  the joystick mix - turning, arcs, turn in place.
#   4  stage 1's speeds WITH the joystick's turning: arcs, turn in place, stop.
#   5  stage 4 with twice the stopping and turning on the spot.
#
# A stage is chosen per run, not promoted by the trainer, and it is passed when `eval_walk` says so:
# 10 s upright while moving, then 1 m/s, then the turns.
STAGES = {
    # **0.15-0.35 m/s, not 0.3-0.8.** At the faster range a BOUND is the cheapest way for this
    # body to reach the commanded speed, and the flight penalty cannot fix that: sweeping it from
    # 1.0 to 3.0 only moved the policy between two attractors, lunging and standing still, with no
    # value in between producing a walk. At a slow amble a step is achievable without leaving the
    # ground, so walking wins on its own merits instead of having to be forced.
    1: dict(forward=(0.15, 0.35), lateral=(0.0, 0.0), turn=(0.0, 0.0),
            mix=(0.10, 0.90, 0.0, 0.0)),
    2: dict(forward=(-0.3, 1.0), lateral=(-0.25, 0.25), turn=(0.0, 0.0),
            mix=(0.10, 0.0, 0.0, 0.90)),
    3: dict(forward=(-0.3, 1.0), lateral=(-0.25, 0.25), turn=(-1.0, 1.0),
            mix=(0.10, 0.30, 0.20, 0.40)),
    # **Turning at the speeds the gait already has.** Stage 3 adds turning AND the full speed range
    # at once, and the full range is where a bound beats a walk (see stage 1). This adds only the
    # turn - arcs, turning on the spot, stopping - at an amble. 0.40, not 0.35, so eval_walk's arcs
    # (0.4 m/s at 0.6 rad/s) are inside what was trained.
    4: dict(forward=(0.15, 0.40), lateral=(0.0, 0.0), turn=(-0.8, 0.8),
            mix=(0.10, 0.30, 0.20, 0.40)),
    # **The two manoeuvres stage 4 did not teach.** After two stage-4 sessions the arcs turned and
    # the straight line held, but told to stand the body still walked away (4 m in 40 s) and told
    # to turn on the spot it barely rotated. Each had 10% and 20% of the commands; this doubles
    # stopping and gives turning on the spot a quarter, out of the straight and arc shares.
    5: dict(forward=(0.15, 0.40), lateral=(0.0, 0.0), turn=(-0.8, 0.8),
            mix=(0.20, 0.25, 0.25, 0.30)),
}


@dataclass(frozen=True)
class WalkCommandConfig:
    forward: tuple[float, float] = CMD_FORWARD
    lateral: tuple[float, float] = CMD_LATERAL
    turn: tuple[float, float] = CMD_TURN
    mix: tuple[float, float, float, float] = (MIX_STAND, MIX_STRAIGHT, MIX_TURN_IN_PLACE, MIX_ARC)

    def __post_init__(self):
        for bounds in (self.forward, self.lateral, self.turn):
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError("Command ranges must contain finite ordered bounds")
        if self.forward[0] == self.forward[1]:
            raise ValueError("The forward range must have positive width for arc sampling")
        if (len(self.mix) != 4 or not all(math.isfinite(v) and v >= 0 for v in self.mix)
                or not math.isclose(sum(self.mix), 1.0)):
            raise ValueError("Command mixture must contain four non-negative probabilities summing to one")

    @classmethod
    def for_stage(cls, stage: int):
        return cls(**STAGES[stage])
