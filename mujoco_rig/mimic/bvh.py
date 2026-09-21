"""Small strict BVH reader for the licensed 100STYLE motion source."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BVH:
    names: list[str]
    parents: list[int]
    offsets: np.ndarray
    channels: list[list[str]]
    frames: np.ndarray
    dt: float

    @classmethod
    def load(cls, path: Path):
        hierarchy, motion = path.read_text().split("MOTION", 1)
        tokens = iter(re.findall(r"[^\s{}]+|[{}]", hierarchy))
        names, parents, offsets, channels = [], [], [], []

        def expect(value):
            actual = next(tokens)
            if actual != value:
                raise ValueError(f"Expected {value}, found {actual}")

        def joint(parent, name):
            index = len(names)
            names.append(name)
            parents.append(parent)
            expect("{")
            expect("OFFSET")
            offsets.append([float(next(tokens)) for _ in range(3)])
            expect("CHANNELS")
            channels.append([next(tokens) for _ in range(int(next(tokens)))])
            while True:
                kind = next(tokens)
                if kind == "}":
                    break
                if kind == "JOINT":
                    joint(index, next(tokens))
                elif kind == "End":
                    expect("Site")
                    expect("{")
                    expect("OFFSET")
                    for _ in range(3):
                        float(next(tokens))
                    expect("}")
                else:
                    raise ValueError(f"Unsupported BVH token {kind}")

        expect("HIERARCHY")
        expect("ROOT")
        joint(-1, next(tokens))
        lines = motion.strip().splitlines()
        count = int(lines[0].split(":")[1])
        dt = float(lines[1].split(":")[1])
        frames = np.loadtxt(lines[2:], ndmin=2)
        if frames.shape != (count, sum(map(len, channels))) or not np.isfinite(frames).all() or dt <= 0:
            raise ValueError("Invalid BVH frame data")
        return cls(names, parents, np.asarray(offsets) * 0.01, channels, frames, dt)

    def world_poses(self, start=0, stop=None):
        frames = self.frames[start:stop]
        positions = np.zeros((len(frames), len(self.names), 3))
        rotations = np.zeros((len(frames), len(self.names), 3, 3))
        cursor = 0
        for j, channels in enumerate(self.channels):
            translation = np.broadcast_to(self.offsets[j], (len(frames), 3)).copy()
            rotation = np.broadcast_to(np.eye(3), (len(frames), 3, 3)).copy()
            for channel in channels:
                values = frames[:, cursor]
                cursor += 1
                if channel.endswith("position"):
                    translation[:, "XYZ".index(channel[0])] += values * 0.01
                elif channel.endswith("rotation"):
                    rotation = rotation @ Rotation.from_euler(channel[0], values[:, None], degrees=True).as_matrix()
                else:
                    raise ValueError(f"Unsupported channel {channel}")
            parent = self.parents[j]
            if parent < 0:
                positions[:, j], rotations[:, j] = translation, rotation
            else:
                positions[:, j] = positions[:, parent] + np.einsum("nij,nj->ni", rotations[:, parent], translation)
                rotations[:, j] = rotations[:, parent] @ rotation
        return positions, rotations
