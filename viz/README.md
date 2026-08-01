# CFRTP process-physics visualizations

Visual companions to the Abaqus/Python CFRTP seeds. Same constitutive models
(Nakamura crystallization, generalized-Maxwell + WLF viscoelasticity, mixed-mode
Benzeggagh–Kenane cohesive) — rendered for talks, the thesis, and advisor
meetings. **Magnitudes are illustrative (uncalibrated); the shapes are the physics.**

## Interactive (drag / play, theme-aware)

`cfrtp_process_physics.html` — a self-contained page with three live modules
(melt→cool→residual stress, elastic vs. viscoelastic, delamination front) plus
an end-to-end pipeline. Open it locally, or use the hosted artifact:

- **https://claude.ai/code/artifact/b29c6bc0-e534-4715-9a1a-952926b68b22**
  (private by default — share from the page's share menu).

## Animated GIFs (drop straight into slides / the thesis)

| file | shows |
|---|---|
| `cfrtp_crystallization_residual.gif` | melt→cool: crystallinity α develops (Nakamura), residual σ₁₁ builds |
| `cfrtp_viscoelastic_relaxation.gif` | elastic vs. viscoelastic over a cure cycle (~relaxation %) |
| `cfrtp_delamination_front.gif` | mixed-mode cohesive front sweeping across the interface |

Each GIF has a matching final-frame `.png` for static figures.

## Regenerate

```
python3 viz/gen_viz_anim.py     # writes viz/*.gif and viz/*.png
```

Pure matplotlib + Pillow (no ffmpeg). Edit the model parameters at the top of
`sim_process` / `sim_cure` / `sdeg_field` in `gen_viz_anim.py`; they mirror the
UMAT constants in `../abaqus/`.
