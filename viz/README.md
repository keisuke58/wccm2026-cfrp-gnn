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

`denso_plate_viewer.html` — 3-D view of an injection-moulded short-fibre plate for the
Denso residual-strain theme: fibres sampled to match the through-thickness orientation
tensor (skin / shell / core), the orientation-tensor ellipsoid at a chosen depth,
through-thickness crystallinity, orientation-averaged CTE and residual strain, warpage
for symmetric vs asymmetric mould temperatures, and an exact split of the residual
strain into its thermal (orientation-driven) and crystallisation parts. Data from
`denso_plate_features.py` (validated kernels; illustrative constants; orientation
profile assumed, not simulated). Hosted artifact:

- **https://claude.ai/artifact/UxNvMza6Swiihe9XjZUM6e** (private by default).

Rebuild: `python3 viz/denso_plate_features.py`, then substitute
`viz/denso_plate_features.json` for `__DATA__` in `denso_plate_viewer.template.html`.

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
