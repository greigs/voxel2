"""Generate a tiny multi-color .vox model for quickly testing the tile pipeline.

Produces ``test.vox``: a 2x2x2 cube plus a 1-wide 2-tall stick on top. Small enough to
process in a couple of seconds, but it still exercises convex edges, convex corners, a
slim "stick" feature, and several paint color-codes.
"""

import argparse

from pyvox.models import Vox, Model
from pyvox.writer import VoxWriter


def main():
    parser = argparse.ArgumentParser(description="Write a tiny test .vox model.")
    parser.add_argument("-o", "--output", default="test.vox", help="Output path (default: test.vox).")
    args = parser.parse_args()

    # Palette (r, g, b, a); voxel color indices are 1-based into this list.
    colors = [
        (220, 60, 60, 255),    # 1 red
        (70, 180, 70, 255),    # 2 green
        (70, 110, 220, 255),   # 3 blue
        (235, 200, 60, 255),   # 4 yellow
    ]
    palette = colors + [(80, 80, 80, 255)] * (255 - len(colors))

    voxels = []
    # 2x2x2 cube (minus the (1,1,1) corner -> adds a concave notch), color varied by
    # position so several color-codes appear.
    for x in range(2):
        for y in range(2):
            for z in range(2):
                if (x, y, z) == (1, 1, 1):
                    continue
                c = 1 + ((x + y + z) % 4)
                voxels.append((x, y, z, c))
    # A 1x1 stick rising out of one corner (slim feature).
    voxels.append((0, 0, 2, 2))
    voxels.append((0, 0, 3, 2))

    size = (2, 2, 4)
    model = Model(size=size, voxels=voxels)
    VoxWriter(args.output, Vox(models=[model], palette=palette)).write()
    print(f"Wrote {len(voxels)} voxels to '{args.output}' (size {size}).")


if __name__ == "__main__":
    main()
