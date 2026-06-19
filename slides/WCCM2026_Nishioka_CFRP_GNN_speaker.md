# Speaker script — WCCM2026_Nishioka_CFRP_GNN.tex

*EN narration extracted from \note{}. Add 和訳 under each slide by hand if wanted.*

---

## Slide 1

Good morning. I am Keisuke Nishioka, from Keio University and Leibniz Universitaet Hannover, with co-authors at Nagoya University and JAXA. Today I show how we localize hidden delamination in perforated CFRP interstage structures using finite-element analysis and a graph neural network: 61% macro-F1 from surface stress alone, and how we make it robust to measurement noise. About 15 minutes.


---

## Slide 2

Quick roadmap: motivation and the DSPSS measurement, the method, the published results, the new noise-robustness work, and outlook. About 15 minutes.


---

## Slide 3

State the three contributions up front, then dive in.


---

## Slide 4

CFRP is the primary structural material in reusable launch vehicles, for its strength-to-weight ratio. Around bolt holes, interlaminar delamination accumulates and is invisible from the outside. Conventional NDT — ultrasonic, X-ray CT, tap testing — is slow, expensive, and operator-dependent. So we need an automated, reliable way to localize these sub-surface defects.


---

## Slide 5

Our measurable quantity is infrared stress measurement. By the thermoelastic effect, a small temperature change is proportional to the change in the sum of principal stresses. So an infrared camera gives a full-field surface map of the stress sum, which we call DSPSS. A sub-surface defect perturbs this field and creates a local stress valley, as shown on the right.


---

## Slide 6

Kojima et al. gave a CNN proof-of-concept on non-perforated CFRP. Our setting is harder: perforated interstage, full 3-D location (region times layer), and extreme imbalance. Formally we learn a node classifier on the mesh graph, minimizing focal loss; class zero is defect-free, classes one to eighteen encode region times layer.


---

## Slide 7

Two anchors, both in Composites B. Kojima 2025 is the direct surface-stress lineage we build on (and a co-author). PIGMID 2026 is an adjacent physics-informed GNN, but on guided-wave sensor signals — a different modality. We differentiate on input, structure, the physics prior, and the OOD generalization study.


---

## Slide 8

We build the dataset by finite-element analysis: a one-eightieth-scale curved interstage panel, about 2 metres radius, with two rectangular holes. Delaminations are modeled explicitly by inserting thin Teflon sheets between selected plies, under tensile loading. This gives DSPSS fields with exactly known ground-truth defects.


---

## Slide 9

The raw DSPSS is dominated by the stress concentration around the holes, which masks the defect. So we subtract a defect-free reference to get a difference field, and z-score normalize per node. This isolates the defect signal and, importantly, improves discrimination in the depth direction — that is, which layer the defect is in.


---

## Slide 10

Looking closer, each defect produces a stress valley at its centre. The depth and width of that valley depend on the ply angle above it: ninety-degree plies give sharp narrow valleys, zero and forty-five broaden them. This ply-angle signature is exactly the cue the network uses to tell which layer the delamination is in. It is a fine spatial-frequency problem, which motivates local attention.


---

## Slide 11

One quick but important point before the network. Detection and localization are different jobs. Detection — is there a defect, and roughly where — needs no learning at all. Because every specimen shares one mesh geometry, the healthy fields form a point cloud around a single template, so the same per-node z-score we already use as input doubles as an unsupervised detector: score each node by how many standard deviations it sits from the healthy mean, with mean and std estimated from two thousand defect-free fields. On clean data this per-node statistic reaches node-level AUROC of about one — it actually beats every learned anomaly detector we tried, including a diffusion model. The one caveat is signal-to-noise, but it is more forgiving than folklore suggests: on a labelled coupon the detector still scores about 0.84 AUROC at noise one-tenth of the field standard deviation, and only falls to chance near seven-tenths to one times the field std — so the often-quoted collapse at sigma equals 0.1 is wrong. Lock-in infrared thermography, where noise falls as one over root-K frames, only adds margin. So screening is cheap. The GNN earns its keep on the genuinely hard part: the semantic nineteen-class region-by-layer identification under noise — which is the rest of this talk.


---

## Slide 12

We represent each surface element as a node, with features being its coordinates and the normalized stress. Edges are the fixed mesh connectivity, reused across all specimens. We use a graph attention network: three GAT layers, sixty-four to two-hundred-fifty-six channels, multi-head, with batch-norm, dropout and residual connections. Attention lets each node weight its neighbours adaptively, which suits the highly non-uniform stress field.


---

## Slide 13

We formulate localization as node classification into nineteen classes: class zero is defect-free, and the other classes jointly encode the in-plane region and the insertion layer. We label only one side, so prediction needs a single surface measurement. Note the severe imbalance — defect nodes are under three percent of all nodes.


---

## Slide 14

To handle that imbalance we use focal loss, which down-weights the easy, abundant defect-free nodes and focuses learning on the rare defect classes. Without it, the model simply predicts defect-free everywhere.


---

## Slide 15

On held-out test specimens we get a planar score R of 0.72, a depth score of 0.69, and a combined TDPS of 0.70, up to 0.92 in the best case. The mid-thickness layers, nine to twelve, are localized best. Critically, the false-negative rate on defect nodes is under three percent, so we rarely miss a real defect.


---

## Slide 16

Overall macro-F1 is sixty-one percent across all nineteen classes, from surface DSPSS alone. The confusion matrix has a strong diagonal; the residual errors are mostly confusion with the adjacent layer, and false positives cluster near the hole edge, where stress gradients are steep.


---

## Slide 17

To summarize the published work: an FEM-plus-GAT framework localizes 3-D delamination in perforated CFRP; difference-normalized DSPSS removes the hole concentration and enables full-region estimation; and we reach sixty-one percent macro-F1 from surface data alone, with very low false negatives and no false detections in healthy regions.


---

## Slide 18

Now the new part for this congress. The published model assumes clean FEM fields. Real infrared measurements carry noise — temperature drift, emissivity, sensor noise. And in practice almost the entire structure is healthy, so false alarms are very costly. So here we focus on robustness under realistic noise and on suppressing false detection. On the right, the input with and without injected noise — the defect signature still survives.


---

## Slide 19

We do two things. First, defect-free negatives: we add five thousand samples that are a zero field plus Gaussian noise at ten percent of the data standard deviation, so the model explicitly learns what no-defect looks like. Second, we add structured noise to the defect data — line noise along the fibre direction, attenuated noise across it, and global white noise — to mimic measurement artefacts.


---

## Slide 20

The outcome: with the difference-DSPSS input, the model still estimates defects across the full region; prediction stays accurate on noise-injected data; and the defect-free negatives remove spurious activations, so we get essentially no false detection on healthy inputs. Here, the noisy input on the left still yields the correct target localization on the right. This makes the framework robust enough to move toward real experimental data.


---

## Slide 21

Going forward: first, sim-to-real — applying this noise-robust model to actual infrared stress measurements. Second, and still preliminary, we are extending from attention GNNs to mesh-physics GNNs such as MeshGraphNet and some layered variants; early results are encouraging and under evaluation with our co-authors. And third, transferring the approach to payload-fairing structural health monitoring. I will stop here. Thank you — I am happy to take questions.


---

## Slide 22

Thank the co-authors and JAXA, then invite questions.


---
