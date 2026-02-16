from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass

from reportlab.lib.colors import black, red
from reportlab.lib.pagesizes import letter as LETTER
from reportlab.pdfgen import canvas

CANONICAL_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class PuzzleSpec:
    """User-selected puzzle definition."""

    alphabet: tuple[str, ...]
    target: tuple[str, ...]
    height: int
    width: int

    @property
    def span(self) -> int:
        return len(self.target)


def in_bounds(spec: PuzzleSpec, r: int, c: int) -> bool:
    """Return whether (r, c) is inside the grid."""
    return 0 <= r < spec.height and 0 <= c < spec.width


def build_segments(spec: PuzzleSpec) -> list[tuple[tuple[int, int], ...]]:
    """Build all length-N segments in 4 canonical directions."""
    segments = []
    for r in range(spec.height):
        for c in range(spec.width):
            for dr, dc in CANONICAL_DIRS:
                cells = tuple((r + k * dr, c + k * dc) for k in range(spec.span))
                if all(in_bounds(spec, rr, cc) for rr, cc in cells):
                    segments.append(cells)
    return segments


def build_cell_to_segments(
    segments: list[tuple[tuple[int, int], ...]],
) -> dict[tuple[int, int], list[int]]:
    """Index each cell to the segment ids that pass through it."""
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, seg in enumerate(segments):
        for cell in seg:
            index[cell].append(i)
    return index


def violates_local(grid: list[list[str]], spec: PuzzleSpec, r: int, c: int) -> bool:
    """Check whether a recent cell edit created target/reverse in nearby segments."""
    reverse = spec.target[::-1]
    for dr, dc in CANONICAL_DIRS:
        for offset in range(-spec.span + 1, 1):
            cells = tuple((r + (offset + k) * dr, c + (offset + k) * dc) for k in range(spec.span))
            if all(in_bounds(spec, rr, cc) for rr, cc in cells):
                word = tuple(grid[rr][cc] for rr, cc in cells)
                if word == spec.target or word == reverse:
                    return True
    return False


def step(grid: list[list[str]], spec: PuzzleSpec, rng: random.Random, lazy_prob: float) -> None:
    """Single-site Metropolis-like update with lazy steps and local rejection."""
    if rng.random() < lazy_prob:
        return

    r = rng.randrange(spec.height)
    c = rng.randrange(spec.width)
    old = grid[r][c]
    new = rng.choice([ch for ch in spec.alphabet if ch != old])

    grid[r][c] = new
    if violates_local(grid, spec, r, c):
        grid[r][c] = old


def run_chain(
    spec: PuzzleSpec,
    rng: random.Random,
    steps: int,
    burn: int,
    thin: int,
    lazy_prob: float,
    n_snapshots: int,
) -> list[list[str]]:
    """Sample valid background pages that avoid target/reverse entirely."""
    # Start from an obviously valid state with a single repeated symbol.
    grid = [[spec.alphabet[0]] * spec.width for _ in range(spec.height)]
    snapshots: list[list[str]] = []

    for t in range(1, steps + 1):
        step(grid, spec, rng, lazy_prob)
        if t >= burn and (t - burn) % thin == 0:
            snapshots.append(["".join(row) for row in grid])
            if len(snapshots) == n_snapshots:
                break

    return snapshots


def count_target_in_affected(
    grid: list[list[str]],
    spec: PuzzleSpec,
    segments: list[tuple[tuple[int, int], ...]],
    cell_to_segments: dict[tuple[int, int], list[int]],
    affected_cells: list[tuple[int, int]],
) -> int:
    """Count target/reverse only in segments touched by changed cells."""
    reverse = spec.target[::-1]
    seen = set()
    total = 0

    for cell in affected_cells:
        for seg_idx in cell_to_segments[cell]:
            if seg_idx in seen:
                continue
            seen.add(seg_idx)
            seg = segments[seg_idx]
            word = tuple(grid[rr][cc] for rr, cc in seg)
            if word == spec.target or word == reverse:
                total += 1

    return total


def inject_exactly_one_target(
    grid: list[list[str]],
    spec: PuzzleSpec,
    rng: random.Random,
    segments: list[tuple[tuple[int, int], ...]],
    cell_to_segments: dict[tuple[int, int], list[int]],
    max_tries: int,
) -> tuple[list[list[str]], dict[str, object]]:
    """Force exactly one target occurrence by editing one candidate segment."""
    patterns = (spec.target, spec.target[::-1])

    for _ in range(max_tries):
        cells = rng.choice(segments)
        pattern = rng.choice(patterns)

        new_grid = [row[:] for row in grid]
        changed: list[tuple[int, int]] = []

        for (rr, cc), ch in zip(cells, pattern):
            if new_grid[rr][cc] != ch:
                new_grid[rr][cc] = ch
                changed.append((rr, cc))

        if not changed:
            continue

        # The base page has zero target hits, so new hits must touch a changed cell.
        if count_target_in_affected(new_grid, spec, segments, cell_to_segments, changed) == 1:
            dr = cells[1][0] - cells[0][0]
            dc = cells[1][1] - cells[0][1]
            info = {
                "cells": cells,
                "dir": (dr, dc),
                "pattern": "".join(pattern),
            }
            return new_grid, info

    raise RuntimeError("Failed to inject exactly one target. Increase --max-inject-tries.")


def write_grids_pdf(
    filename: str,
    grids: list[list[str]],
    target_label: str,
    marks: list[dict[str, object] | None] | None = None,
) -> None:
    """Render puzzle pages and optional highlighted answer cells to a PDF."""
    doc = canvas.Canvas(filename, pagesize=LETTER)
    page_w, page_h = LETTER
    margin_x = 72
    margin_top = 60
    margin_bottom = 60
    grid_top = page_h - margin_top - 40
    grid_bottom = margin_bottom + 30
    avail_h = grid_top - grid_bottom
    avail_w = page_w - 2 * margin_x

    for i, rows in enumerate(grids):
        red_cells = set()
        if marks is not None and marks[i] is not None:
            red_cells = set(marks[i]["cells"])

        n_rows = len(rows)
        n_cols = len(rows[0]) if n_rows else 0
        fs_h = avail_h / (n_rows * 1.35)
        fs_w = avail_w / (n_cols * 1.30)
        font_size = max(8, min(18, min(fs_h, fs_w) * 1.15))

        font_name = "Courier"
        doc.setFont(font_name, font_size)
        char_w = doc.stringWidth("M", font_name, font_size)
        tracking = char_w * 0.55
        pitch = char_w + tracking
        line_h = font_size * 1.25

        grid_w = pitch * (n_cols - 1) + char_w
        grid_h = line_h * (n_rows - 1) + font_size
        start_x = (page_w - grid_w) / 2
        start_y = grid_bottom + (avail_h + grid_h) / 2

        doc.setFillColor(black)
        doc.setFont("Times-Roman", 12)
        doc.drawCentredString(page_w / 2, page_h - margin_top, f"Find exactly one {target_label}")

        y = start_y
        doc.setFont(font_name, font_size)
        for r, row in enumerate(rows):
            x = start_x
            for c, ch in enumerate(row):
                doc.setFillColor(red if (r, c) in red_cells else black)
                doc.drawString(x, y, ch)
                x += pitch
            y -= line_h

        doc.setFillColor(black)
        doc.setFont("Times-Roman", 10)
        doc.drawCentredString(page_w / 2, margin_bottom / 2, str(i + 1))
        doc.showPage()

    doc.save()


def make_book(
    spec: PuzzleSpec,
    n_pages: int,
    steps: int | None,
    burn: int,
    thin: int,
    lazy_prob: float,
    seed: int,
    out_pdf: str,
    out_key_pdf: str,
    answer_key: bool,
    fox_page: int | None,
    max_inject_tries: int,
) -> dict[str, object]:
    """Generate pages, inject target on one page, and write output PDFs."""
    rng = random.Random(seed)
    segments = build_segments(spec)
    if not segments:
        raise ValueError("Grid is too small for the target length; no valid segments exist.")
    cell_to_segments = build_cell_to_segments(segments)

    min_steps = burn + (n_pages - 1) * thin
    if steps is None:
        # Auto-size steps so we can always collect the requested number of snapshots.
        steps = min_steps
    if steps < min_steps:
        raise ValueError(f"--steps must be at least {min_steps} to produce {n_pages} pages.")

    pages = run_chain(
        spec=spec,
        rng=rng,
        steps=steps,
        burn=burn,
        thin=thin,
        lazy_prob=lazy_prob,
        n_snapshots=n_pages,
    )
    if len(pages) < n_pages:
        raise ValueError(f"Only generated {len(pages)} snapshots; expected {n_pages}.")

    if fox_page is None:
        page_index = rng.randrange(n_pages)
    else:
        if fox_page < 1 or fox_page > n_pages:
            raise ValueError("--fox-page must be between 1 and --pages.")
        page_index = fox_page - 1

    selected = [list(row) for row in pages[page_index]]
    injected, info = inject_exactly_one_target(
        grid=selected,
        spec=spec,
        rng=rng,
        segments=segments,
        cell_to_segments=cell_to_segments,
        max_tries=max_inject_tries,
    )

    pages_with_target = list(pages)
    pages_with_target[page_index] = ["".join(row) for row in injected]
    target_label = "".join(spec.target)

    write_grids_pdf(filename=out_pdf, grids=pages_with_target, target_label=target_label, marks=[None] * n_pages)

    if answer_key:
        marks = [None] * n_pages
        marks[page_index] = info
        write_grids_pdf(filename=out_key_pdf, grids=pages_with_target, target_label=target_label, marks=marks)

    return {"target_page_index": page_index, "target_info": info}


def parse_symbols(raw: str) -> tuple[str, ...]:
    """Parse alphabet from 'FOX' or comma-separated 'F,O,X' forms."""
    text = raw.strip()
    if not text:
        raise ValueError("Alphabet cannot be empty.")
    symbols = [s.strip().upper() for s in text.split(",") if s.strip()] if "," in text else [
        ch.upper() for ch in text if not ch.isspace()
    ]
    return tuple(symbols)


def parse_word(raw: str) -> tuple[str, ...]:
    """Parse target word symbols, ignoring spaces/commas."""
    letters = tuple(ch.upper() for ch in raw if not ch.isspace() and ch != ",")
    if not letters:
        raise ValueError("Target word cannot be empty.")
    return letters


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate printable 'Find the Fox' puzzle PDFs.")
    parser.add_argument("--alphabet", default="FOX", help="Three letters used to fill the grid, e.g. FOX or F,O,X")
    parser.add_argument("--target", default="FOX", help="Three-letter word to hide exactly once.")
    parser.add_argument("--height", type=int, default=32, help="Number of grid rows.")
    parser.add_argument("--width", type=int, default=20, help="Number of grid columns.")
    parser.add_argument("--pages", type=int, default=1, help="Number of puzzle pages.")
    parser.add_argument("--steps", type=int, default=None, help="MCMC steps. Auto-sized if omitted.")
    parser.add_argument("--burn", type=int, default=50_000, help="Burn-in steps.")
    parser.add_argument("--thin", type=int, default=50_000, help="Steps between snapshots.")
    parser.add_argument("--lazy-prob", type=float, default=0.5, help="Stay-put probability per MCMC step.")
    parser.add_argument("--seed", type=int, default=1729, help="Random seed.")
    parser.add_argument("--out-pdf", default="find_the_fox.pdf", help="Puzzle PDF output path.")
    parser.add_argument(
        "--out-key-pdf",
        default="find_the_fox_answer_key.pdf",
        help="Answer-key PDF output path.",
    )
    parser.add_argument("--answer-key", dest="answer_key", action="store_true", help="Generate answer key PDF.")
    parser.add_argument("--no-answer-key", dest="answer_key", action="store_false", help="Skip answer key PDF.")
    parser.set_defaults(answer_key=True)
    parser.add_argument("--fox-page", type=int, default=None, help="1-based page to place target on.")
    parser.add_argument(
        "--max-inject-tries",
        type=int,
        default=200_000,
        help="Attempts when forcing one target instance.",
    )
    return parser.parse_args()


def validate_inputs(
    spec: PuzzleSpec,
    n_pages: int,
    steps: int | None,
    burn: int,
    thin: int,
    lazy_prob: float,
    max_inject_tries: int,
) -> None:
    positive_int("--height", spec.height)
    positive_int("--width", spec.width)
    positive_int("--pages", n_pages)
    positive_int("--thin", thin)
    non_negative_int("--burn", burn)
    positive_int("--max-inject-tries", max_inject_tries)
    if steps is not None:
        positive_int("--steps", steps)
    if len(spec.alphabet) != 3:
        raise ValueError("Alphabet must contain exactly 3 letters.")
    if len(set(spec.alphabet)) != 3:
        raise ValueError("Alphabet must contain 3 different letters.")
    if len(spec.target) != 3:
        raise ValueError("Target word must contain exactly 3 letters.")
    if len(set(spec.target)) != 3:
        raise ValueError("Target word must contain 3 different letters.")
    missing = sorted(set(spec.target) - set(spec.alphabet))
    if missing:
        raise ValueError(f"Target contains symbols not in alphabet: {', '.join(missing)}")
    if not 0.0 <= lazy_prob <= 1.0:
        raise ValueError("--lazy-prob must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    try:
        spec = PuzzleSpec(
            alphabet=parse_symbols(args.alphabet),
            target=parse_word(args.target),
            height=args.height,
            width=args.width,
        )
        validate_inputs(
            spec=spec,
            n_pages=args.pages,
            steps=args.steps,
            burn=args.burn,
            thin=args.thin,
            lazy_prob=args.lazy_prob,
            max_inject_tries=args.max_inject_tries,
        )

        result = make_book(
            spec=spec,
            n_pages=args.pages,
            steps=args.steps,
            burn=args.burn,
            thin=args.thin,
            lazy_prob=args.lazy_prob,
            seed=args.seed,
            out_pdf=args.out_pdf,
            out_key_pdf=args.out_key_pdf,
            answer_key=args.answer_key,
            fox_page=args.fox_page,
            max_inject_tries=args.max_inject_tries,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote puzzle PDF: {args.out_pdf}")
    if args.answer_key:
        print(f"Wrote answer key PDF: {args.out_key_pdf}")
    print(f"Target inserted on page: {result['target_page_index'] + 1}")
    print(f"Target info: {result['target_info']}")


if __name__ == "__main__":
    main()
