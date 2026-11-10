"""Per-frame fitting with a warm start: a scan over frames, a fold per frame.

The shape of a body fit over the frames of a video (ai_vision body3d): the
state is the pose fitted on the previous frame, each frame starts its fit
from it and gives an output of its own. Here the "pose" is a point and the
"fit" gradient descent towards the frame's observed point:

- ``IcoScan`` over the frames: ``Pose → Iterator[tuple[FrameFit, Pose]]``,
  the step ``tuple[Frame, Pose] → tuple[FrameFit, Pose]``;
- inside the step, ``IcoFold`` over the iterations of the optimizer with
  ``until`` on a loss plateau: a warm start needs few iterations;
- a stream after the scan keeps the outputs and drops the states.

Run: ``uv run python src/examples/ico_tracking.py``
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace

from ico import IcoFold, IcoOperator, IcoScan, IcoSource, IcoStream

MAX_ITERATIONS = 200
PATIENCE = 3  # iterations without a better loss


@dataclass(frozen=True)
class Frame:
    """An observed point on frame ``index``."""

    index: int
    x: float
    y: float


@dataclass(frozen=True)
class Pose:
    """The state between frames: the fitted point."""

    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class FrameFit:
    """The output of a frame: its pose and how many iterations it took."""

    index: int
    pose: Pose
    iterations: int


@dataclass(frozen=True)
class Optimization:
    """The state of one frame's fit: pose, target, loss and a plateau count."""

    pose: Pose
    target: Frame
    iterations: int = 0
    best_loss: float = float("inf")
    stale: int = 0


def frames() -> Iterator[Frame]:
    """A point moving slowly to the right."""
    return (Frame(i, 10.0 + 0.5 * i, 5.0 + 0.1 * i) for i in range(8))


def iterations() -> Iterator[int]:
    return iter(range(MAX_ITERATIONS))


def gradient_step(pair: tuple[int, Optimization]) -> Optimization:
    """One step of gradient descent on the squared distance to the target."""
    _, opt = pair
    pose, target = opt.pose, opt.target
    lr = 0.3
    pose = Pose(pose.x - lr * (pose.x - target.x), pose.y - lr * (pose.y - target.y))
    loss = (pose.x - target.x) ** 2 + (pose.y - target.y) ** 2
    better = loss < opt.best_loss - 1e-4
    return replace(
        opt,
        pose=pose,
        iterations=opt.iterations + 1,
        best_loss=min(loss, opt.best_loss),
        stale=0 if better else opt.stale + 1,
    )


def plateau(opt: Optimization) -> bool:
    return opt.stale >= PATIENCE


optimize = IcoFold(IcoSource(iterations), gradient_step, until=plateau, name="optimize")


def start_fit(pair: tuple[Frame, Pose]) -> Optimization:
    """Start a frame's fit from the pose of the previous one (a warm start)."""
    frame, pose = pair
    return Optimization(pose=pose, target=frame)


def finish_fit(opt: Optimization) -> tuple[FrameFit, Pose]:
    """The frame's output, and its pose as the state for the next frame."""
    return FrameFit(opt.target.index, opt.pose, opt.iterations), opt.pose


# The step of the scan is an ordinary chain; the fold is a node in it.
fit_frame = IcoOperator(start_fit) | optimize | IcoOperator(finish_fit)


def output(pair: tuple[FrameFit, Pose]) -> FrameFit:
    """Keep the frame's output, drop the state."""
    return pair[0]


tracking = IcoScan(IcoSource(frames), fit_frame, name="track") | IcoStream(
    IcoOperator(output)
)


if __name__ == "__main__":
    tracking.describe()
    for fit in tracking(Pose()):
        print(
            f"frame {fit.index}: ({fit.pose.x:.2f}, {fit.pose.y:.2f}) "
            f"in {fit.iterations} iterations"
        )
