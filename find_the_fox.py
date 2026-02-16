import random

letters = ("F", "O", "X")
height, width = 32, 20

# 8 directions (dx, dy)
dirs = [(0,1), (1,0), (1,1), (1,-1),
        (-1,0), (0,-1), (-1,1), (-1,-1)]

def in_bounds(r, c):
    return 0 <= r < height and 0 <= c < width

def violates_local(grid, r, c, directions):
     # check whether FOX appears in any 3-cell segment that includes (r,c), along any direction
    for dr, dc in directions:
        # the triple can be (r-2dr,r-dr,r), (r-dr,r,r+dr), or (r,r+dr,r+2dr)
        for offset in (-2, -1, 0):
            coords = [(r + (offset + k)*dr, c + (offset + k)*dc) for k in range(3)]
            if all(in_bounds(rr, cc) for rr, cc in coords):
                triple = tuple(grid[rr][cc] for rr, cc in coords)
                if triple in (letters, letters[::-1]):
                    return True
    return False

def is_valid(grid):
    # full scan
    for r in range(height):
        for c in range(width):
            if violates_local(grid, r, c, dirs):
                return False
    return True




from collections import defaultdict
from itertools import combinations, product

# generate all length-3 segments
segments = [
    tuple((r + k*dr, c + k*dc) for k in range(3))
    for r in range(height) for c in range(width)
    for dr, dc in dirs[:4]
    if all(in_bounds(r + k*dr, c + k*dc) for k in range(3))
]

# index segments by cell
cell_to_segs = defaultdict(list)
for i, seg in enumerate(segments):
    for cell in seg:
        cell_to_segs[cell].append(i)

# get all overlapping segment pairs
pairs = {
    tuple(sorted(p))
    for idxs in cell_to_segs.values()
    for p in combinations(idxs, 2)
}

# compute probability that two segments are both FOX/XOF
def pair_prob(s1, s2):
    total = 0.0
    for p1, p2 in product((letters, letters[::-1]), repeat=2):
        req = {}
        ok = True
        for (cell, ch) in zip(s1, p1):
            if cell in req and req[cell] != ch:
                ok = False; break
            req[cell] = ch
        for (cell, ch) in zip(s2, p2):
            if cell in req and req[cell] != ch:
                ok = False; break
            req[cell] = ch
        if ok:
            total += 3 ** (-len(req))
    return total

Delta = sum(pair_prob(segments[i], segments[j]) for i, j in pairs)
print("Delta =", Delta)




def step(grid, lazy_p=0.5):
    # one step of the lazy single-site chain; grid is a list of lists
    if random.random() < lazy_p:
        return

    r = random.randrange(height)
    c = random.randrange(width)
    old = grid[r][c]
    new = random.choice([x for x in letters if x != old])

    grid[r][c] = new
    if violates_local(grid, r, c, dirs[:4]):  # reject if we created a forbidden triple
        grid[r][c] = old              # revert

def run_chain(steps=500_000, burn=50_000, thin=50_000, seed=None):
    if seed is not None:
        random.seed(seed)

    # start from an easy valid state (all F)
    grid = [["F"] * width for _ in range(height)]

    snapshots = []
    for t in range(1, steps + 1):
        step(grid)

        if t >= burn and (t - burn) % thin == 0:
            snapshots.append(["".join(row) for row in grid])

    return snapshots


pages = run_chain(seed=1729)
print("\n".join(pages[0][:10]))




segments = []

for r in range(height):
    for c in range(width):
        for dr, dc in dirs[:4]:  # →, ↓, ↘, ↙
            coords = [(r + k*dr, c + k*dc) for k in range(3)]
            if all(in_bounds(rr, cc) for rr, cc in coords):
                segments.append((tuple(coords), (dr, dc)))

def segments_through_cell(r, c):
    # all length-3 segments (4 directions) that include (r,c)
    out = []
    for dr, dc in dirs[:4]:
        for off in (-2, -1, 0):
            coords = [(r + (off+k)*dr, c + (off+k)*dc) for k in range(3)]
            if all(in_bounds(rr, cc) for rr, cc in coords):
                out.append(tuple(coords))
    return out

def count_fox_in_affected(grid, affected_cells):
    # count FOX/XOF occurrences among segments that intersect affected_cells
    seen = set()
    total = 0
    for (r, c) in affected_cells:
        for seg in segments_through_cell(r, c):
            if seg in seen:
                continue
            seen.add(seg)
            triple = tuple(grid[rr][cc] for rr, cc in seg)
            if triple in (letters, letters[::-1]):
                total += 1
    return total

def inject_exactly_one_fox(grid, max_tries=200000):
    for _ in range(max_tries):
        (cells, (dr, dc)) = random.choice(segments)
        pat = random.choice((letters, letters[::-1]))  # choose FOX or XOF orientation on this segment

        new = [row[:] for row in grid]
        changed = []
        for (rr, cc), ch in zip(cells, pat):
            if new[rr][cc] != ch:
                new[rr][cc] = ch
                changed.append((rr, cc))

        # if we didn't change anything, we'd still have 0 occurrences — skip
        if not changed:
            continue

        # starting grid has 0; any new occurrence must touch a changed cell
        if count_fox_in_affected(new, changed) == 1:
            return new, {"cells": cells, "dir": (dr, dc), "pattern": pat}

    raise RuntimeError("Failed to inject exactly one FOX; increase max_tries.")




from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter as LETTER
from reportlab.lib.colors import black, red

def write_grids_pdf(filename, grids, fox_marks=None):
    c = canvas.Canvas(filename, pagesize=LETTER)
    page_w, page_h = LETTER

    # page layout stuff
    margin_x = 72
    margin_top = 60
    margin_bot = 60

    # grid area bounds (leaves room for title and page number)
    grid_top = page_h - margin_top - 40
    grid_bot = margin_bot + 30
    avail_h = grid_top - grid_bot
    avail_w = page_w - 2 * margin_x

    # monospaced font for the grid
    font_name = "Times-Roman"

    for i, rows in enumerate(grids):
        # build lookup for highlighted cells on this page
        red_cells = set()
        if fox_marks is not None and fox_marks[i] is not None:
            red_cells = set(fox_marks[i].get("cells", []))

        n_rows = len(rows)
        n_cols = len(rows[0]) if n_rows else 0
        fs_h = avail_h / (n_rows * 1.35)
        fs_w = avail_w / (n_cols * 1.30)
        font_size = min(fs_h, fs_w) * 1.15
        font_size = max(8, min(18, font_size))

        c.setFont(font_name, font_size)
        char_w = c.stringWidth("M", font_name, font_size)

        tracking = char_w * 0.70     # extra spacing between letters
        pitch = char_w + tracking
        line_h = font_size * 1.25

        grid_w = pitch * (n_cols - 1) + char_w
        grid_h = line_h * (n_rows - 1) + font_size

        start_x = (page_w - grid_w) / 2
        start_y = grid_bot + (avail_h + grid_h) / 2  # near top of grid area

        # draw grid
        y = start_y
        for r, row in enumerate(rows):
            x = start_x
            for cc, ch in enumerate(row):
                c.setFillColor(red if (r, cc) in red_cells else black)
                c.drawString(x, y, ch)
                x += pitch
            y -= line_h

        # page number (bottom center)
        c.setFillColor(black)
        c.setFont("Times-Roman", 10)
        c.drawCentredString(page_w / 2, margin_bot / 2, str(i + 1))

        c.showPage()

    c.save()


def make_book(n_pages=1,
              steps=1_100_000,
              burn=50_000,
              thin=50_000,
              seed=1729,
              out_pdf="find_the_fox.pdf",
              out_key_pdf="find_the_fox_answer_key.pdf"):
    random.seed(seed)

    pages = run_chain(steps=steps, burn=burn, thin=thin, seed=seed)
    if len(pages) < n_pages:
        raise ValueError(f"run_chain produced only {len(pages)} snapshots; need {n_pages}. "
                         "Increase steps or reduce thin.")

    pages = pages[:n_pages]

    # choose which page gets the FOX
    j = random.randrange(n_pages)

    # inject exactly one FOX/XOF on that page
    grid = [list(row) for row in pages[j]]
    new_grid, info = inject_exactly_one_fox(grid)

    pages_with_fox = list(pages)
    pages_with_fox[j] = ["".join(row) for row in new_grid]

    # book PDF: no highlighting
    write_grids_pdf(out_pdf, pages_with_fox, fox_marks=[None]*n_pages)

    # answer key PDF
    marks = [None]*n_pages
    marks[j] = info
    write_grids_pdf(out_key_pdf, pages_with_fox, fox_marks=marks)

    return {"fox_page_index": j, "fox_info": info}


result = make_book(n_pages=1)
print("FOX inserted on page:", result["fox_page_index"] + 1)
print("FOX info:", result["fox_info"])
