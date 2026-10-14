"""Build an Arena-only projectile pool without changing the training model."""

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def build(source: Path, destination: Path, count: int = 64) -> None:
    if count < 1 or source.resolve() == destination.resolve():
        raise ValueError("Use a separate destination and a positive ball count")
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("worldbody")
    template = world.find("body[@name='ball']")
    if template is None or any(b.get("name", "").startswith("ball_") for b in world.findall("body")):
        raise ValueError("Source must contain exactly one appended ball")
    for i in range(1, count):
        ball = deepcopy(template)
        ball.set("name", f"ball_{i}")
        ball.set("pos", f"{40 + .4 * (i % 8):g} {40 + .4 * (i // 8):g} 2")
        ball.find("freejoint").set("name", f"ball_free_{i}")
        for geom in ball.findall("geom"):
            geom.set("name", f"{geom.get('name')}_{i}")
        world.append(ball)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    build(args.source, args.destination, args.count)
