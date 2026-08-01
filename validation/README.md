# B-1 · Method validation on public carbon/PEEK data

The Daikin fluoropolymer-CF system is proprietary, so we validate the **method**
(the crystallization model in `../abaqus/cfrtp_cryst_umat_ve.f`) against the
**documented non-isothermal crystallization behaviour of carbon/PEEK**, for which
public data exist.

```
python3 validation/cfrtp_peek_validation.py   # prints PASS/FAIL + writes the figure
```

## What it checks (universal, documented laws)

Running the same Nakamura law as the UMAT in **physical time** over a sweep of
cooling rates (2–160 °C/min):

| check | law | result |
|---|---|---|
| **V1** | crystallization peak Tp **decreases** with cooling rate | ✅ PASS (338 → 273 °C over 2 → 160 °C/min) |
| **V2** | α(T) curves **shift to lower T** with faster cooling | ✅ PASS |
| **V3** | final crystallinity **non-increasing**, collapses at high rate | ✅ PASS (1.0 → 0.27 at 160 °C/min) |

![PEEK crystallization validation](peek_crystallization_validation.png)

## Finding (why this matters)

The bare bell `K(T) = KMAX·exp(−((T−TCRYST)/WCRYST)²)` used in the UMAT/deck has a
**high-T tail that does not vanish near the melt**: at slow cooling it let PEEK
"crystallize" **above Tm (343 °C)** — unphysical (nucleation needs undercooling).
This harness adds an **undercooling cutoff** (`K = 0 for T ≥ Tm`), after which Tp
stays below Tm and the trends come out right.

**Recommended next step (physics deepening):**
1. Add the same `Tm` cutoff to `cfrtp_cryst_umat_ve.f` (small, safe — needs a
   re-verify run on the Abaqus box), and/or
2. Move `K(T)` to the **Hoffman–Lauritzen** form (transport × nucleation factors,
   vanishing at **both** Tg and Tm) for quantitative Tp. The crude bell reproduces
   the qualitative laws but predicts slow-cool Tp too close to Tm (~338 vs the
   literature PEEK ~305–310 °C).

## Quantitative comparison hook

Drop a digitized reference `validation/peek_reference_Tp.csv` with a header
`rate_C_per_min,Tp_C` (from a paper's DSC figure) and the script auto-computes
`RMSE(Tp)` between model and literature. This is the drop-in point once a dataset
is in hand.

## Notes
- **Time units**: this runs in physical time (°C/min); the deck's `KMAX` is in the
  step's normalized time, a different unit — so the numeric `KMAX` differs by design.
- Magnitudes remain **illustrative** until DMA / measured crystallinity data.

**Refs** (see `../README.md`): Tierney & Gillespie, *Composites Part A* 35 (2004);
"Modeling of PEEK Crystallization Kinetics Under Transient Thermal Conditions,"
*Polymers* (MDPI); Nakamura et al., *J. Appl. Polym. Sci.* 16 (1972).
