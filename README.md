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

---

## Features

- Pure Python implementation
- Constraint-preserving local proposals
- Flexible grid size and target word
- Easily extensible to other constraint types

---
