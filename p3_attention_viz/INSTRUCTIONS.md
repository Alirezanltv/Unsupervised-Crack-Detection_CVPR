# Attention-gate visualization — defends the title

The paper's title promises "Attention-Guided Feature Transfer", and Sec. 3.4
currently states, honestly, that no interpretability claim is made. A
reviewer will push on that. This experiment either earns the claim or tells
you to reframe — both are better than silence.

## Steps
1. Adapt `load_model_and_gates()` in `visualize_gates.py` to your AG-DSCAE
   code (one function; docstring shows the forward-hook pattern).
2. Run on the first 6 test images of each dataset (fixed, not hand-picked):
   `python visualize_gates.py --checkpoint best.pt --images .../test/images --out gates_concrete/`
3. Read `gate_stats.csv`. The informal expectation is
   `gate_on_edges > gate_on_background` at most scales.

## What goes in the paper
- **If the expectation holds**: add `gate_maps.png` (or a curated version —
  say so in the caption) as a supplementary figure, one exemplar row in main
  Sec. 3.4 if the page budget allows, and REPLACE the "no interpretability
  claim" sentence with the measured statement (e.g., "gates average X on
  edge pixels vs. Y on background at scale 1; supplementary Sec. E").
- **If it does not hold**: the figure still goes in the supplement with an
  honest sentence, and consider whether "attention-guided" should soften to
  "gated" in the title. An unexamined titular claim is a bigger risk than a
  nuanced result.

Never publish only the images that look best without saying how they were
selected.
