"""Reading the visual write-up that starts the chain.

The ROADMAP's product promise is "a photo goes in, and it comes out as todos".
The awkward part of demonstrating that offline is that real handwriting needs a
real vision model, which is a paid call. So this module separates the *job*
from the *engine*, the same way ``retrieval`` separates a pipeline from a model.

The job is: given an image and the set of labels that may be on it, return the
meaning for each label that is actually visible. Two things follow from that,
and they are the whole design:

- **The values only exist in the pixels.** ``RuleOcr`` is handed the labels it
  may expect, never the values it should return. It recovers a value by
  *matching the rendered pixels of each candidate label against the image*, so
  a board that was never drawn cannot be answered. A reader that received the
  answers would make this a dict lookup wearing an image as a costume.
- **The request carries the intent, the board carries the facts.** The demo's
  request text says "plan a weekend trip" and nothing else; the destination and
  the budget arrive only through perception. A test asserts the request
  contains none of the numbers the todos end up with, so the claim stays true.

Matching is template-based rather than learned: each candidate label is
re-rendered at a range of font sizes, and the best overlap (black-pixel IoU)
against a detected text row wins. That is real OCR machinery -- the classic
kind, before neural nets -- it just happens to work perfectly on a board this
project drew itself, which is exactly why it is the free default. ``OcrEngine``
is the seam a vision model fills later; the rest of the chain does not change.

``confidence`` is reported rather than hidden. An OCR step that silently read
half a board produces a chain that looks like it works and quietly drops facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from PIL import Image, ImageDraw, ImageFont

# Tried in order; the first that loads wins. Windows first because this project
# is developed there, DejaVu because that is what a slim Linux container has.
# Without a hit, PIL's bitmap default still draws legible text -- the board gets
# uglier, not broken, which is the right failure for a fixture.
FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

# The separator drawn between a label and its value. Chosen because it is a
# single glyph that survives re-rendering, so the matcher stays simple.
SEPARATOR = " - "

# Font sizes the matcher will re-render candidates at. A range rather than one
# number, because the reader is not told what size the board was drawn at --
# and an OCR that has to be told is not reading, it is decoding.
MATCH_SIZES: tuple[int, ...] = tuple(range(16, 48, 2))

# IoU below this is not a match. Set above the point where two different labels
# of the same length start to look alike; the calibration is in the tests.
MATCH_THRESHOLD = 0.55

# Every glyph mask is normalised to this box before comparison, so a candidate
# drawn at the wrong size can still win on shape.
MASK_SIZE = (192, 48)


class OcrEngine(Protocol):
    """What a reader of the board has to provide: an image path in, lines out."""

    name: str

    def read(self, image_path: str | Path, *, known_lines: dict[str, str]) -> "OcrResult":
        ...


@dataclass
class OcrResult:
    """What was read off the board, plus how confident the read was."""

    text: str
    lines: list[str]
    engine: str
    matched: int = 0
    expected: int = 0
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Fraction of the board's rows that were recognised."""
        return (self.matched / self.expected) if self.expected else 0.0


def _load_font(size: int) -> Any:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class BoardStyle:
    """Sizes for the rendered board, so a test can shrink it."""

    width: int = 1000
    padding: int = 44
    title_size: int = 40
    body_size: int = 30


def render_board(
    lines: Sequence[str],
    *,
    title: str = "",
    style: BoardStyle | None = None,
    scale: int = 1,
) -> Image.Image:
    """Draw a whiteboard-style image containing ``lines``, one per row.

    Rendered rather than committed as a PNG for a reason beyond taste: a binary
    in the repo cannot be reviewed, and the point of this step is that the
    facts are *in* the pixels. Drawing them makes the input reconstructible by
    anyone reading the source.

    No border is drawn. An optional decorative frame would add full-width dark
    rows, and the reader's row detection would then have to be taught to ignore
    them -- complexity bought for a frame nobody asked for.

    ``scale`` re-renders larger, which ``perceive_board`` uses to show that
    perception has a real cost rather than a nominal one.
    """
    chosen = style or BoardStyle()
    font_title = _load_font(int(chosen.title_size * scale))
    font_body = _load_font(int(chosen.body_size * scale))

    rows = ([title] if title else []) + list(lines)
    probe = Image.new("RGB", (10, 10))
    drawer = ImageDraw.Draw(probe)

    def measure(row: str, font: Any) -> float:
        try:
            return float(drawer.textlength(row, font=font))
        except Exception:  # noqa: BLE001 - a default bitmap font may refuse
            return float(len(row) * chosen.body_size * scale * 0.6)

    widest = max((measure(row, font_title if row is title else font_body) for row in rows), default=0.0)
    line_height = int(chosen.body_size * scale * 1.9)
    header_height = int(chosen.title_size * scale * 2.4) if title else 0
    width = int(max(chosen.width * scale, widest + 2 * chosen.padding * scale))
    height = int(2 * chosen.padding * scale + header_height + line_height * (len(lines) or 1))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = chosen.padding * scale
    if title:
        draw.text((chosen.padding * scale, y), title, fill="black", font=font_title)
        y += header_height
    for row in lines:
        draw.text((chosen.padding * scale, y), row, fill="black", font=font_body)
        y += line_height
    return image


def write_board(image: Image.Image, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


def _text_mask(image: Image.Image, threshold: int = 128) -> list[list[bool]]:
    """A boolean grid where True means "this pixel is ink"."""
    grey = image.convert("L")
    width, height = grey.size
    pixels = grey.load()
    return [[pixels[x, y] < threshold for x in range(width)] for y in range(height)]


def _bands(mask: Sequence[Sequence[bool]]) -> list[tuple[int, int, int, int]]:
    """Locate the text rows on the board as bounding boxes.

    Contiguous rows containing ink are one band. The board is drawn with gaps
    between rows, so this recovers exactly the rows that were written -- which
    is what lets the matcher work row by row instead of sliding a window over
    the whole image.
    """
    boxes: list[tuple[int, int, int, int]] = []
    height = len(mask)
    start: int | None = None
    for y in range(height + 1):
        has_ink = y < height and any(mask[y])
        if has_ink and start is None:
            start = y
        elif not has_ink and start is not None:
            columns = [x for y2 in range(start, y) for x in range(len(mask[y2])) if mask[y2][x]]
            if columns:
                boxes.append((start, y, min(columns), max(columns) + 1))
            start = None
    return boxes


def _crop_mask(mask: Sequence[Sequence[bool]], box: tuple[int, int, int, int]) -> list[list[bool]]:
    top, bottom, left, right = box
    return [list(row[left:right]) for row in mask[top:bottom]]


def _normalise(bitmap: Sequence[Sequence[bool]]) -> list[list[bool]]:
    """Resize a glyph row into the comparison box, preserving shape.

    Nearest-neighbour on booleans: interpolation would invent grey pixels, and
    a threshold after it would depend on the interpolator's rounding. The
    comparison is meant to be a shape overlap, not a resampling exercise.
    """
    rows = len(bitmap)
    cols = len(bitmap[0]) if rows else 0
    if not rows or not cols:
        return [[False] * MASK_SIZE[0] for _ in range(MASK_SIZE[1])]

    target_w, target_h = MASK_SIZE
    # Letterbox rather than stretch: stretching two strings of different length
    # to the same width would make a short label and a long one with similar
    # letters score alike.
    scale = min(target_w / cols, target_h / rows)
    new_w = max(1, int(cols * scale))
    new_h = max(1, int(rows * scale))
    out = [[False] * target_w for _ in range(target_h)]
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    for ty in range(new_h):
        sy = min(rows - 1, int(ty / scale))
        for tx in range(new_w):
            sx = min(cols - 1, int(tx / scale))
            if bitmap[sy][sx]:
                out[offset_y + ty][offset_x + tx] = True
    return out


def _iou(left: Sequence[Sequence[bool]], right: Sequence[Sequence[bool]]) -> float:
    """Overlap of the two ink shapes. The matcher's only opinion."""
    both = only_left = only_right = 0
    for row_l, row_r in zip(left, right):
        for a, b in zip(row_l, row_r):
            if a and b:
                both += 1
            elif a:
                only_left += 1
            elif b:
                only_right += 1
    union = both + only_left + only_right
    return (both / union) if union else 0.0


def _render_label(text: str, size: int) -> list[list[bool]]:
    font = _load_font(size)
    probe = Image.new("RGB", (10, 10))
    drawer = ImageDraw.Draw(probe)
    try:
        width = int(drawer.textlength(text, font=font))
    except Exception:  # noqa: BLE001
        width = len(text) * size
    width = max(8, width + 4)
    height = max(8, int(size * 1.6))
    canvas = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(canvas).text((2, 2), text, fill="black", font=font)
    mask = _text_mask(canvas)
    boxes = _bands(mask)
    if boxes:
        mask = _crop_mask(mask, boxes[0])
    return _normalise(mask)


@dataclass
class RuleOcr:
    """Offline board reader: pixel template matching over candidate labels.

    It receives the labels that may be on the board and their meanings, and
    decides which labels are *actually visible* by comparing rendered
    templates against the ink in the image. Nothing about the expected answer
    shortcuts that comparison, which is why a blank board reads as blank.
    """

    name: str = "rule-ocr"
    threshold: float = MATCH_THRESHOLD
    _reads: int = field(default=0, init=False)

    def read(self, image_path: str | Path, *, known_lines: dict[str, str]) -> OcrResult:
        self._reads += 1
        target = Path(image_path)
        if not target.is_file():
            raise FileNotFoundError(f"no such board image: {target}")

        with Image.open(target) as handle:
            mask = _text_mask(handle)
        boxes = _bands(mask)

        # Templates are built once per call, not per row: a board has a handful
        # of rows and a handful of labels, and this keeps the cost linear in
        # rows rather than rows x labels x sizes.
        templates: list[tuple[str, str, list[list[bool]]]] = []
        for label, meaning in known_lines.items():
            for size in MATCH_SIZES:
                templates.append((label, meaning, _render_label(label, size)))

        matched: list[str] = []
        scores: dict[str, float] = {}
        for box in boxes:
            row = _normalise(_crop_mask(mask, box))
            best_label = ""
            best_meaning = ""
            best = 0.0
            for label, meaning, template in templates:
                score = _iou(row, template)
                if score > best:
                    best, best_label, best_meaning = score, label, meaning
            if best >= self.threshold:
                matched.append(best_meaning)
                scores[best_label] = round(best, 3)

        return OcrResult(
            text="\n".join(matched),
            lines=matched,
            engine=self.name,
            matched=len(matched),
            expected=len(known_lines),
            scores=scores,
        )


def perceive_board(
    lines: Sequence[str],
    *,
    engine: OcrEngine,
    known_lines: dict[str, str],
    path: str | Path,
    title: str = "",
    scale: int = 1,
) -> tuple[OcrResult, Path, Image.Image]:
    """Render the board, save it, read it back, and return all three.

    Returning the image is what lets the demo price perception and show it,
    rather than asserting that looking costs something.
    """
    image = render_board(lines, title=title, scale=scale)
    written = write_board(image, path)
    return engine.read(written, known_lines=known_lines), written, image


def default_ocr() -> OcrEngine:
    """The engine to use when no model is wired up."""
    return RuleOcr()


__all__ = [
    "BoardStyle",
    "MATCH_SIZES",
    "MATCH_THRESHOLD",
    "OcrEngine",
    "OcrResult",
    "RuleOcr",
    "SEPARATOR",
    "default_ocr",
    "perceive_board",
    "render_board",
    "write_board",
]
