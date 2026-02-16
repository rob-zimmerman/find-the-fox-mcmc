# 🦊 Find the Fox with MCMC

A Python implementation of a Markov Chain Monte Carlo (MCMC) sampler for generating random word-search grids conditioned on containing a specified hidden word.

This project accompanies the blog post:

https://rob-zimmerman.github.io/posts/2026-find-the-fox

---

## Overview

Suppose we want to generate a random $m \times n$ grid of letters, uniformly at random, subject to the constraint that the grid contains a specific word (e.g., "FOX") somewhere inside it.

Naïve rejection sampling becomes infeasible as the grid grows, since the probability of containing the target word becomes small.

Instead, we construct a Markov chain on the space of valid grids and use MCMC to sample from the conditional distribution.

The algorithm:

1. Starts from a valid grid containing the target word.
2. Proposes local modifications to the grid.
3. Accepts moves that preserve the constraint.
4. Iterates to approximate samples from the uniform distribution over valid grids.

The repo:
- generates printable word-search style puzzle PDFs
- every page is sampled to avoid the target word (and its reverse) in all rows/columns/diagonals
- exactly one page has exactly one injected target word
- an optional answer key highlights the target letters

---

## Requirements

- Python 3.10+
- `reportlab`

Install dependency:

```bash
pip install reportlab
```

## Quick Start

Generate a single-page puzzle and answer key:

```bash
python find_the_fox.py
```

Generate 10 pages (auto-computes the minimum number of MCMC steps):

```bash
python find_the_fox.py --pages 10
```

Generate puzzle only (no answer key):

```bash
python find_the_fox.py --pages 10 --no-answer-key
```

## Useful Options

- `--alphabet`: symbols used to fill the grid. Examples: `FOX` or `F,O,X`
- `--target`: word to hide exactly once (default: `FOX`)
- `--height`, `--width`: grid size
- `--pages`: number of pages
- `--seed`: reproducible random seed
- `--fox-page`: force the target onto a specific page (1-based)
- `--out-pdf`: puzzle output path
- `--out-key-pdf`: answer-key output path
- `--burn`, `--thin`, `--steps`, `--lazy-prob`: MCMC controls

Show all flags:

```bash
python find_the_fox.py --help
```

## Examples

Use a different alphabet/target and custom grid:

```bash
python find_the_fox.py \
  --alphabet C,A,T \
  --target CAT \
  --height 28 \
  --width 28 \
  --pages 5 \
  --seed 42 \
  --out-pdf find_the_cat.pdf \
  --out-key-pdf find_the_cat_key.pdf
```

Use explicit MCMC steps:

```bash
python find_the_fox.py --pages 8 --burn 50000 --thin 40000 --steps 330000
```
