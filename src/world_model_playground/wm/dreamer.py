"""Converting episodic collections into DreamerV3-style step sequences.

DreamerV3 consumes one record per environment step (``T+1`` per episode)
rather than transitions: ``image``, ``reward`` received on arriving at the
step, ``is_first``, ``is_last``, ``is_terminal``, and ``action``. Ports differ
in which action a step carries, so it is selectable:

- ``"preceding"``: the action that led to the step, zeros on the first step
  (the convention of the PyTorch port ``dreamerv3-torch``).
- ``"following"``: the action taken from the step, zeros on the last step
  (the convention of the original JAX ``dreamerv3`` replay buffer).
"""

from collections.abc import Mapping
from typing import Literal

import numpy as np

from world_model_playground.wm.episodes import validate_episode

ActionAlignment = Literal["preceding", "following"]
ACTION_ALIGNMENTS: tuple[ActionAlignment, ...] = ("preceding", "following")


def to_dreamer_episode(
    episode: Mapping[str, np.ndarray],
    *,
    image_key: str,
    action_alignment: ActionAlignment = "preceding",
) -> dict[str, np.ndarray]:
    """Returns the ``T+1`` step arrays of one episode in DreamerV3's layout.

    ``state`` carries the low-dimensional observation, for proprioceptive or
    mixed encoders; ``discount`` is ``1 - is_terminal`` for ports that read it.
    """
    length = validate_episode(episode)
    if image_key not in episode:
        raise ValueError(f"episode has no {image_key}; render it at collection time")
    actions = np.asarray(episode["actions"], dtype=np.float32)
    no_action = np.zeros_like(actions[:1])
    if action_alignment == "preceding":
        step_actions = np.concatenate((no_action, actions))
    elif action_alignment == "following":
        step_actions = np.concatenate((actions, no_action))
    else:
        raise ValueError(f"action_alignment must be one of {ACTION_ALIGNMENTS}")

    is_first = np.zeros(length + 1, dtype=np.bool_)
    is_first[0] = True
    is_last = np.zeros(length + 1, dtype=np.bool_)
    is_last[-1] = True
    is_terminal = is_last & bool(episode["terminated"])
    return {
        "image": np.asarray(episode[image_key], dtype=np.uint8),
        "state": np.asarray(episode["observations"], dtype=np.float32),
        "action": step_actions,
        "reward": np.concatenate(([0.0], episode["rewards"])).astype(np.float32),
        "discount": (~is_terminal).astype(np.float32),
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }
