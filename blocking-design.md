# DARE: Depth-Adaptive Residual Embeddings for Blocking
 
*A design document for an anytime, generation-bound vector-blocking system for entity resolution.*
 
> **Working codename:** DARE (Depth-Adaptive Residual Embedding). Rename freely.
> **Status:** design / pre-implementation. Intended as the basis for a code repo and an empirical paper.
> **One-line thesis:** Co-design the embedding *generator* and the *search* so that a query record is encoded incrementally, deepest-only-when-needed, with a learned per-step estimate of the information still in the tail — and integrate this tightly enough into the index that "compute more" reuses prior activations instead of recomputing.
 
---
 
## 1. Motivation and scope
 
### 1.1 The problem
 
Master Data Management (MDM) and entity resolution (ER) pipelines must avoid the O(n²) cost of comparing every record to every other record. The standard remedy is **blocking**: a cheap stage that produces a small set of candidate pairs, which an expensive **matcher** then adjudicates. A modern blocking style concatenates a record's fields, embeds the result, and runs vector search to retrieve candidates (the "DeepBlocker" pattern). This handles dirty, heterogeneous data well but is expensive on two fronts:
 
1. **Generation cost** — running a full encoder (768–1024 dim transformer) per record.
2. **Search cost** — comparing high-dimensional vectors across a large corpus.
DARE targets the **generation-bound regime specifically**: the encoder is the bottleneck, and after deterministic predicates the candidate set per query is small. This is the regime where most ANN research (which assumes vectors are precomputed and stored) does *not* optimize, and where adaptive generation can win.
 
### 1.2 What makes blocking a favorable target
 
Blocking is uniquely forgiving in a way that reshapes the whole design:
 
- **Ambiguity is not failure.** A near-tie the early computation cannot resolve is exactly a record that should be *forwarded as a candidate* for the matcher to settle. The blocker is allowed to be uncertain; it is only forbidden from **dropping a true match**.
- This inverts the stopping criterion. We do not stop when confident in a *winner*; we stop when confident in a **partition** — when the candidate set has split cleanly into "clearly in" and "clearly out," with a small ambiguous middle that we forward wholesale.
- The operative quality metric is therefore **recall at a given forwarded-candidate-set size**, not top-1 accuracy. The only true errors are confident "clearly out" calls on real matches; spurious "clearly in" calls merely inflate the set the matcher cleans up.
### 1.3 Design principles (the thread that produced this)
 
- **Adaptivity is asymmetric.** The query side is generated lazily and incrementally; the database side is fully precomputed at every depth. One deeper query pass refines the comparison against the *entire* candidate set at once.
- **Vectorize the residual, not the determined.** Fields consumed by deterministic blocking predicates (and anything they functionally determine) are stripped from the encoder input, so embedding capacity and vector variance are spent on real residual differences — which widens margins and makes early stopping safer.
- **Predict the tail, don't just bound it.** A learned head reads the shared trunk activations and estimates the magnitude of the effect remaining chunks will have. Wrap it in a conformal/risk-control layer so the resulting guarantee is statistically valid, not merely heuristic.
- **Build-and-measure.** The contribution is empirical: a working system, a real benchmark suite, strong tuned baselines, and a characterization of the speed/recall/storage frontier in this regime.
---
 
## 2. Theoretical background and related work
 
DARE is a recombination of several mature lines of work. The novelty is the *join* — lazy generation + residual chunking + learned valid stopping, integrated as a blocking index in the generation-bound regime — not any single component. Knowing the prior art precisely is necessary both for positioning and for borrowing the right tools.
 
### 2.1 Ordered / nested representations
 
- **Matryoshka Representation Learning (MRL)** — Kusupati et al., NeurIPS 2022. Trains one encoder with a loss summed over nested prefixes so the earliest dimensions carry the most information; prefixes are independently usable. Shipped in OpenAI `text-embedding-3`, Nomic Embed, others. *DARE borrows: the prefix-ordering property.*
- **2D Matryoshka Sentence Embeddings (2DMSE)** — arXiv 2402.14776. Extends nesting to **two axes**: embedding width *and* transformer depth, so intermediate layers yield usable embeddings. Crucial observation for us: plain MRL still requires traversing all layers, so it reduces storage/compare cost but **not generation cost**; the depth axis is the one that reduces generation. *DARE borrows: the depth axis as the generation-cost lever.*
- **Starbucks** (follow-up to 2DMSE) — trains a fixed ladder of (layer, dim) pairs and finds that **separately/explicitly trained sub-models outperform the implicitly nested sub-networks** of 2DMSE at matched sizes, at higher training cost. *Implication for DARE: explicit per-chunk supervision is worth the training expense.*
- **MatFormer** — nested transformer along the width/FFN axis; another granularity dial.
### 2.2 Adaptive-computation generation
 
- **Adaptive Computation Time (ACT)** — Graves, 2016; **PonderNet** — Banino et al., 2021. Learn *when to halt* recurrent computation.
- **BranchyNet** — Teerapittayanon et al., 2017; **early-exit BERT / DeeBERT / PABEE**; **LayerSkip** — Elhoushi et al., ACL 2024. Intermediate classifiers/readouts enable per-input depth. *DARE borrows: depth early-exit, but for embedding readout driving a search, not classification.*
- **Autoregressive coarse-to-fine generation** — FlexTok (tail-drop tokenizer forcing information into earlier tokens), DetailFlow, Spectral Image Tokenizer (interruptible coarse decode), CausalEmbed (Matryoshka ordering emerges from autoregressive training), CoFiRec (recommendation). *Relevant as the "emit coordinates one chunk at a time, stop early" family; DARE uses depth chunks rather than a sequential decoder to avoid per-coordinate latency.*
### 2.3 Energy-ordered / partial-distance search
 
- **Partial Distortion / Partial Distance Search (PDS)** — classic vector-quantization literature (e.g., Bei & Gray, 1985). Accumulate distance dimension-by-dimension; prune a candidate the moment its partial distance exceeds the current best. **Exact.** *DARE borrows: incremental distance accumulation with pruning.*
- **PCA / energy ordering** — leading components by variance; computable incrementally (power iteration, NIPALS). The interpretable ancestor of learned ordering.
- **Panorama** (2025) — learns orthogonal transforms compacting >90% of signal energy into the first half of dimensions, then prunes on partial distances with **no recall loss** (2–30× speedup). *The closest published system to DARE's search side, but operates on stored vectors with exact bounds and no lazy generation.*
- **AdANNS** — Rege, Kusupati et al., NeurIPS 2023. Uses Matryoshka representations of different capacities at different ANNS pipeline stages (coarse shortlist, fine rerank); up to ~16× faster at matched accuracy. *DARE borrows: coarse-to-fine retrieval, extended to the generation side.*
### 2.4 Nearest neighbor as best-arm identification (the statistical core)
 
This is the formalization of "compute some coordinates, get a distance estimate with uncertainty, compute more only for candidates still in contention."
 
- **Adaptive Estimation for Approximate k-NN** — LeJeune, Heckel, Baraniuk, AISTATS 2019. Reduces k-NN to a **multi-armed bandit** (LUCB / Hamming-LUCB): each candidate is an arm, sampling coordinates yields noisy distance estimates, maintain confidence bounds, identify the close/far sets with high confidence.
- **Adaptive Monte-Carlo Optimization / Bandit-Based MC for NN** — Bagaria, Kamath, Tse, 2018; **Medoids via MAB** — Bagaria et al., AISTATS 2018. Cost measured as the **fraction of coordinates evaluated** — exactly DARE's currency.
- **Nearest Neighbor Search Under Uncertainty** — Mason, Tripathy, Nowak, UAI 2021; **Learning NN Graphs from Noisy Distance Samples** — NeurIPS 2019.
- **Empirical Bernstein bounds** — Maurer & Pontil, 2009. Variance-adaptive confidence intervals — the principled version of "if the residual variance is small, you can stop." *DARE replaces these distribution-free bounds with a learned, sharper head, then re-validates with conformal methods.*
- **Best-arm identification** background — Audibert, Bubeck, Munos (Successive Rejects), COLT 2010; Jamieson & Nowak. **Covariance-adaptive BAI** (2023) — exploits correlation among arms; relevant because candidate distances are highly correlated in a learned space.
### 2.5 Calibrated / statistically valid early exit
 
The "is the probability real?" problem — and it is known to be hard.
 
- **Rethinking Calibration for Early-Exit NNs (EEFP)** — 2025 (arXiv 2508.21495): confidence **calibration alone is insufficient**; intermediate exits are systematically overconfident. Predict *failure*, not just confidence.
- **Conditional monotonicity for anytime classification** — Jazbec et al., NeurIPS 2023; **Fast yet Safe: Early-Exiting with Risk Control** — Jazbec et al., NeurIPS 2024.
- **Confident Adaptive Language Modeling (CALM)** — Schuster et al., NeurIPS 2022; **Confident Adaptive Transformers** — EMNLP 2021.
- **Conformal prediction / conformal risk control** — Vovk et al.; Angelopoulos & Bates. *DARE borrows: distribution-free finite-sample guarantees layered over the learned head.*
### 2.6 Entity resolution and blocking
 
- **DeepBlocker** — Thirumuruganathan et al., VLDB 2021. Self-supervised embedding-based blocking; the pattern DARE accelerates.
- **Magellan** — Konda et al., VLDB 2016; **DeepMatcher** — Mudgal et al., SIGMOD 2018. Standard ER toolkits and benchmark sources.
- **ZeroER** — Wu et al., SIGMOD 2020 (uses cheap overlap blocking before the expensive stage — precedent for layering predicates and embeddings).
- **Rule-based / GPU blocking** — e.g., HyperBlocker; blocking surveys by Papadakis et al. *DARE borrows: deterministic predicates as the first filter, then residual-only vector blocking.*
### 2.7 Quantization and vector databases (baseline territory)
 
- **Product Quantization** — Jégou et al., PAMI 2011; **residual / additive quantization**; **FAISS** — Johnson et al. (residual quantizer with refinement is the "more bits on demand" baseline).
- **Vector DBs** — Qdrant (multi-stage prefetch/rerank; ACORN metadata-in-graph filtering), Weaviate, Milvus. The off-the-shelf multi-stage-search primitives; note none generate query embeddings lazily.
---
 
## 3. System architecture
 
### 3.1 Overview
 
```
                    ┌─────────────────────────────────────────────┐
   record ──▶ strip │  Shared trunk (transformer blocks)           │
   (residual fields │                                              │
    only, per       │  block 1 ─▶ block 2 ─▶ block 3 ─▶ ... ─▶ N   │
    blocking        │     │          │          │                  │
    scheme)         │     ▼          ▼          ▼                  │
                    │  chunk e₁    chunk e₂    chunk e₃   (16–64d   │
                    │  + head c₁   + head c₂   + head c₃   each)    │
                    └─────┼──────────┼──────────┼──────────────────┘
                          │          │          │
   query path:  compute chunk, append to running query vector,
                compare against precomputed DB chunks (same depth),
                consult head + margins → STOP or DEEPEN (reuse activations)
 
   DB path:     every record precomputed and stored at ALL depths
                (chunks are concatenable; storage = full vector + per-depth norms)
```
 
### 3.2 The encoder: shared trunk, depth-incremental residual chunks
 
- A single backbone. Computation proceeds **block by block**; deeper blocks consume the cached activations of shallower ones, so advancing one step is **incremental, not a restart**. This is the core efficiency claim and the thing a cascade of independent models cannot do.
- After block *k*, a lightweight **readout head** emits embedding chunk `e_k ∈ R^{d_chunk}` (e.g. 16–64 dims). The depth-`k` embedding is the concatenation `E_k = [e_1; …; e_k]`.
- **Residual training (central):** `e_k` is trained on what `E_{k-1}` failed to separate. Concretely, supervise `e_k` to reduce the residual ranking error on pairs that `E_{k-1}` leaves ambiguous, and **decorrelate** `e_k` from `E_{k-1}` (penalize cross-chunk covariance). This forces later chunks to carry the discriminative structure that distinguishes near-duplicates — the property that motivates stepping deep enough to reach them.
- **Per-chunk whitening / scale normalization** so that concatenated chunks combine into a meaningful distance (independent-chunk outputs otherwise have incompatible scales).
### 3.3 The confidence / value head
 
After block *k*, a second head `c_k` reads the trunk activations (which already contain the latent information later chunks will project out) and predicts a **distribution over the residual magnitude** `‖E_∞ − E_k‖` (or, better, its effect on decision margins — see §3.5). Key design commitments:
 
- **Predict the decision effect, not the embedding delta.** A large embedding shift that preserves the ranking is irrelevant; a tiny shift that flips the top two is fatal. Factor the head into (a) an offline, query-only **residual-distribution predictor** and (b) an online propagation of that distribution through the **current candidate margins** to get `P(top decision flips with more depth)`.
- **Asymmetric loss.** Underestimating a true pair's similarity (risking a dropped match) is penalized far more than overestimating. This matches the blocker's recall-first objective and is an easier learning target than precise magnitude.
- **Sufficient depth.** The head is a real sub-network, not a single linear layer; it must be expressive enough to read tail magnitude from the trunk state.
- **The residual is observable, the mapping must be learned.** The information the tail will use is present in the trunk activations, so this is not an observability barrier — it is a *coverage* problem. The head is reliable where the activation→tail-effect mapping resembles training and unreliable in structurally rare regions (the unusual near-duplicate). The fix is data (hard-pair mining, §4.3), not capacity, plus the conformal floor (§3.6) for the residual rare miss.
### 3.4 Per-blocking-scheme residual conditioning
 
- Strip fields consumed by deterministic predicates from the encoder input. Within a "same-day" block the date is constant, so embedding it only compresses margins; removing it widens the gaps the head measures.
- Generalize beyond literal predicate fields to anything **functionally determined** by the blocking key (e.g., city given zip): such fields carry little residual signal and mostly add redundancy.
- Strongest form: **train the encoder per blocking scheme** on in-block pairs, so it learns the residual distribution each predicate leaves behind. This is also a clean novelty hook (the vectorizer is specialized to residual structure, which general-purpose embedders are not).
### 3.5 Stopping rule: partition confidence, not winner confidence
 
At each depth, for the current candidate set:
 
1. Compute the query chunk; extend the running query vector; rescore candidates (incrementally — add the chunk's contribution to the running distance, PDS-style).
2. Use `c_k` + current margins to estimate, for each candidate, `P(its in/out classification flips with more depth)`.
3. Move candidates with low flip-probability into **clearly-in** or **clearly-out**. Keep the rest ambiguous.
4. **Stop when** the ambiguous set is small enough to forward (target candidate-set size reached) *and* the predicted probability of any clearly-out candidate being a true match is below the risk level δ.
5. Otherwise **deepen** (run the next block from cached activations) and repeat.
6. Output = clearly-in ∪ ambiguous, forwarded to the matcher.
### 3.6 Validity wrapper
 
- The learned head gives a **sharp point estimate**; conformal prediction / risk control converts it into a **statistically valid** rule using a held-out calibration set, with a tunable recall guarantee.
- Framing for the paper: *learned head for sharpness, conformal wrapper for validity.* The wrapper is not a hedge against a weak head — it is the cheap formal guarantee (one calibration set, zero architecture change) that lets us *claim* the head's quality with a bounded miss rate on the rare tail.
### 3.7 Database integration
 
- **Indexing:** for every record, compute the deterministic blocking key(s) and store in a filterable field; compute and store the full embedding (all chunks) plus per-depth cumulative norms (for exact PDS-style bounds where wanted).
- **Query:** resolve the blocking key → fetch the (small) in-block candidate set (ACORN-style filtering inside the index if available) → run the adaptive-depth loop, comparing query chunks against the precomputed DB chunks at matching depth.
- **Why it pays:** query-side generation savings scale with throughput (every incoming record is a query); DB-side cost is amortized at index time. The deeper the encoder and the smaller the block, the better the economics.
---
 
## 4. Training
 
### 4.1 Objectives
 
Per depth `k`, combine:
 
- **Ranking / contrastive loss** on `E_k` (e.g. InfoNCE or supervised contrastive over labeled match/non-match pairs), summed across depths (MRL-style) so every prefix is usable.
- **Residual supervision** on `e_k`: target the pairs `E_{k-1}` leaves ambiguous; reward `e_k` for separating them.
- **Decorrelation penalty:** minimize cross-chunk covariance `Cov(e_k, E_{k-1})` so chunks carry complementary, not redundant, signal.
- **Whitening / scale regularization** per chunk.
- **Head loss:** asymmetric regression/quantile loss on the realized residual effect (predicted vs. actual margin change at full depth), with heavier penalty for under-estimating true-pair similarity.
### 4.2 Schedule (stability matters)
 
The residual/head target moves while the encoder trains, and the auxiliary loss can degrade the primary embedding. Recommended:
 
1. **Phase 1:** train the encoder + chunk readouts (ranking + residual + decorrelation), with `stop_gradient` on the residual targets to keep them from chasing themselves.
2. **Phase 2:** freeze (or slow-LR) the encoder; fit the confidence heads `c_k` on realized residual effects.
3. Optional **Phase 3:** light joint fine-tune with small head-loss weight.
### 4.3 Hard-pair mining (closes the coverage gap)
 
Because the head's errors concentrate in structurally rare near-duplicates, actively mine these: pairs that are identical on early chunks but separated only at depth, cross-block hard negatives, and synthetic near-duplicates (controlled perturbations of single residual fields). Residual-aware sampling — oversample pairs with large `E_∞ − E_k` — directly targets the head's blind spots.
 
### 4.4 Per-scheme training
 
Train (or LoRA-adapt) a variant per blocking scheme on in-block pairs. Evaluate whether per-scheme specialization beats a single general residual encoder (it is both a hypothesis and an ablation, §6).
 
---
 
## 5. The chunk-width / step-stride hypothesis
 
A central, explicitly-empirical question, stated so it can be falsified.
 
- **Bias–variance of chunk width.** Wide chunks → stable per-step distance estimates, coarse stopping granularity, fewer head calls. Narrow chunks → finer stopping dial but noisier per-step estimates (fewer dims to average), which *widen* the per-step CI and can make early stopping *less* willing to fire.
- **The residual-stride hypothesis (the user's intuition, sharpened).** With explicit residual training, later chunks are enriched for the discriminative signal that separates near-duplicates. A flat granularity that compares only shallow chunks can therefore *under-sample the separating signal* for 95%-similar records. Stepping ≥2 blocks per comparison (e.g. train at 16-wide granularity, compare at 32-wide effective windows) may surface those differences sooner. **This is the distinct mechanism residual training adds beyond MRL variance-ordering, and is a primary thing the experiments must test.**
- **Decoupling.** Training granularity (how finely chunks are supervised) and comparison stride (how many blocks per stopping decision) are separable knobs. Sweep both.
Hypothesis to test: *the optimal comparison window is wider than the training-chunk width (variance stabilization), while fine training granularity still helps because it concentrates discriminative signal into reachable late chunks.*
 
---
 
## 6. Experimental design
 
### 6.1 Datasets
 
- **Standard ER benchmarks** (from Magellan/DeepMatcher): structured and dirty/textual — e.g. DBLP–ACM, DBLP–Scholar, Amazon–Google, Walmart–Amazon, Abt–Buy, Fodors–Zagat. Mix of clean and noisy.
- **A residual-heavy / blocking-stress case:** a dataset with strong deterministic predicates available (date, zip, category) so the per-scheme residual-conditioning idea (§3.4) can be exercised; if no public one fits, construct a semi-synthetic one with controlled near-duplicates.
- **A general retrieval set** (e.g. an MTEB retrieval task) to show the encoder is not overfit to ER and the adaptive machinery generalizes.
### 6.2 Baselines (the make-or-break of an empirical paper — tune them hard)
 
- **Brute force** full-dim encoder + exact search (correctness ceiling, speed floor).
- **Single nested model** (2DMSE / early-exit) with fixed-depth readout — *the strongest direct competitor.*
- **MRL small→large cascade** (e.g. Qdrant prefetch-rerank) with full query generation.
- **FAISS residual-quantization + refinement** (the "more bits on demand" line).
- **Panorama** (energy-ordered partial-distance, exact) and **AdANNS** (adaptive ANN) — the closest search-side systems.
- **DeepBlocker** as the domain blocking baseline.
### 6.3 Metrics
 
- **Primary:** recall @ forwarded-candidate-set size (the blocking-correct metric), as a frontier against compute.
- **Compute / latency:** encoder FLOPs and wall-clock per query (the generation-bound axis); end-to-end including search.
- **Storage:** index footprint (per-depth chunks + norms).
- **Tail recall:** recall on the *ambiguous slice* as a function of stopping aggressiveness — **the headline figure.** The approach is most likely to fail here, so the evidence must be strongest here. Aggregate recall that looks good while tail recall frays = hidden failure a reviewer will assume.
- **Head calibration:** predicted tail-effect vs. realized rank-flip, sliced by ambiguity. Tight even on the hard slice ⇒ the head is doing real work and the conformal floor rarely binds.
### 6.4 Ablations
 
- Chunk width ∈ {8, 16, 32, 64} × comparison stride ∈ {1, 2, 4} (tests §5).
- Confidence head on/off (vs. distribution-free empirical-Bernstein stopping from the bandit literature).
- Conformal wrapper on/off (sharpness vs. validity trade).
- Field-stripping on/off; per-scheme vs. general encoder (tests §3.4).
- Residual/decorrelation loss on/off (does explicit residual training actually enrich late chunks?).
- Predict-vs-measure stopping: learned head (no lookahead tax) vs. compute-one-more-chunk-and-measure (Richardson/Runge-Kutta-style), vs. hybrid (measure only near threshold).
### 6.5 Stopping-rule study (a self-contained contribution)
 
Put all stopping rules on the *same* trunk and compare their speed/recall frontiers: single-increment (RK-style), sequence-extrapolation (Aitken/Richardson on the increment series, using increment variance), learned head, distribution-free CI, each ± conformal wrapper. No one has done this head-to-head for adaptive embedding retrieval.
 
---
 
## 7. Risks and mitigations
 
| Risk | Where it bites | Mitigation |
|---|---|---|
| **Novelty positioning** | Reviewers cite bandit-NN / Panorama / early-exit | Frame as empirical system in the *generation-bound, residual-blocking* regime none of them target; cite all explicitly; lead with the build + benchmark. |
| **Head coverage gap** | Rare near-duplicates (structurally unusual) | Hard-pair mining (§4.3); conformal floor; rely on blocking's tolerance for forwarding ambiguous cases. |
| **Heuristic vs. valid** | "Is the probability real?" | Conformal / risk-control wrapper with held-out calibration and a tunable recall guarantee. |
| **Training instability** | Moving residual target, aux loss degrades embedding | Two-phase schedule with stop-gradient; careful loss weighting. |
| **Hardware throughput** | Small candidate sets, SIMD/GPU eats full dot products cheaply | Restrict claims to the generation-bound regime; report the crossover; lean on deep encoders + small blocks where adaptivity wins. |
| **Chunks don't compose** | Independent/late chunks have incompatible scales or redundant info | Decorrelation penalty + per-chunk whitening; verify with the residual-loss ablation. |
| **Lookahead tax** | Measuring the next increment costs a stage | Learned head predicts without computing; hybrid only measures near threshold. |
 
---
 
## 8. Suggested repository structure
 
```
dare/
├── README.md
├── dare/
│   ├── model/
│   │   ├── trunk.py            # shared backbone, block-wise forward w/ activation cache
│   │   ├── readout.py          # per-depth chunk heads + whitening
│   │   ├── confidence.py       # per-depth value/uncertainty heads
│   │   └── losses.py           # ranking + residual + decorrelation + asymmetric head loss
│   ├── data/
│   │   ├── er_benchmarks.py     # Magellan/DeepMatcher loaders
│   │   ├── field_stripping.py   # predicate-aware input construction
│   │   └── hard_pairs.py        # residual-aware / near-duplicate mining
│   ├── index/
│   │   ├── store.py            # per-depth chunk storage + cumulative norms
│   │   ├── adaptive_search.py  # depth-incremental loop, PDS accumulation, stopping
│   │   └── stopping.py         # head-based, bandit-CI, RK, sequence-extrapolation, conformal
│   ├── calibrate/
│   │   └── conformal.py        # held-out calibration → valid recall guarantee
│   └── eval/
│       ├── metrics.py          # recall@candset-size, tail recall, calibration curves
│       └── baselines/          # 2DMSE, MRL-cascade, FAISS-RQ, Panorama, AdANNS, DeepBlocker
├── experiments/
│   ├── configs/                # chunk width × stride × stopping-rule sweeps
│   └── run.py
└── tests/
```
 
**Stack:** PyTorch for the encoder/heads; HuggingFace for the backbone init; FAISS and/or Qdrant for baseline search and the production index; standard ER benchmark loaders (py_entitymatching / DeepMatcher data). Keep the adaptive search loop in a tight, profiled module — its overhead must not eat the generation savings.
 
### Milestones
 
1. **M0 — harness:** datasets, field-stripping, metrics, brute-force + single-model baselines. (Establishes the frontier to beat.)
2. **M1 — encoder:** shared trunk + chunk readouts + residual/decorrelation losses; verify nested prefixes are usable and late chunks separate hard pairs (residual-loss ablation).
3. **M2 — adaptive search:** depth-incremental loop with a *fixed-depth oracle* stopping rule; measure the speed/recall frontier vs. baselines.
4. **M3 — confidence head:** train heads; calibration curves; head-vs-bandit-CI vs. measure-based stopping.
5. **M4 — validity:** conformal wrapper; recall-guarantee knob; tail-recall headline figure.
6. **M5 — sweeps + paper:** chunk/stride study, per-scheme study, stopping-rule study; write-up.
---
 
## 9. Summary
 
DARE encodes a query record **incrementally by depth**, reusing activations across steps, emitting **residual-trained embedding chunks** whose later pieces concentrate the signal that separates near-duplicates, and attaches a **learned, asymmetric confidence head** that estimates the decision-relevant information still in the tail — all integrated into the index so that "compute more" is cheap and shared across candidates, and wrapped in a **conformal guarantee** so the recall floor is statistically valid. It is purpose-built for the **generation-bound, residual-blocking** regime, where deterministic predicates supply small candidate sets and ambiguity is correctly handled by *forwarding* rather than *deciding*. Every component has prior art; the contribution is the join, the regime, and the measured frontier — exactly the shape of a build-and-benchmark systems paper.
 
---
 
## References (verify details before submission)
 
1. Kusupati et al. *Matryoshka Representation Learning.* NeurIPS 2022.
2. *2D Matryoshka Sentence Embeddings.* arXiv:2402.14776.
3. *Starbucks: layer-and-dimension nested embedding training* (2DMSE follow-up).
4. Devvrit, Kudugunta et al. *MatFormer: Nested Transformer for Elastic Inference.*
5. Graves. *Adaptive Computation Time for Recurrent Neural Networks.* 2016.
6. Banino et al. *PonderNet: Learning to Ponder.* 2021.
7. Teerapittayanon et al. *BranchyNet.* 2017.
8. Elhoushi et al. *LayerSkip.* ACL 2024.
9. Bei & Gray. *An improvement of the minimum distortion encoding algorithm (Partial Distance Search).* 1985.
10. *Panorama: energy-ordered partial-distance exact ANN.* 2025.
11. Rege, Kusupati et al. *AdANNS: A Framework for Adaptive Semantic Search.* NeurIPS 2023.
12. LeJeune, Heckel, Baraniuk. *Adaptive Estimation for Approximate k-Nearest-Neighbor Computations.* AISTATS 2019.
13. Bagaria, Kamath, Tse. *Adaptive Monte-Carlo Optimization.* 2018.
14. Bagaria, Kamath, Ntranos, Zhang, Tse. *Medoids in almost linear time via multi-armed bandits.* AISTATS 2018.
15. Mason, Tripathy, Nowak. *Nearest Neighbor Search Under Uncertainty.* UAI 2021.
16. Mason, Tripathy, Nowak. *Learning Nearest Neighbor Graphs from Noisy Distance Samples.* NeurIPS 2019.
17. Maurer & Pontil. *Empirical Bernstein Bounds and Sample Variance Penalization.* COLT 2009.
18. Audibert, Bubeck, Munos. *Best Arm Identification in Multi-Armed Bandits.* COLT 2010.
19. *Covariance Adaptive Best Arm Identification.* 2023.
20. *Rethinking Calibration for Early-Exit Neural Networks (EEFP).* arXiv:2508.21495, 2025.
21. Jazbec et al. *Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity.* NeurIPS 2023.
22. Jazbec et al. *Fast yet Safe: Early-Exiting with Risk Control.* NeurIPS 2024.
23. Schuster et al. *Confident Adaptive Language Modeling (CALM).* NeurIPS 2022.
24. Schuster et al. *Consistent Accelerated Inference via Confident Adaptive Transformers.* EMNLP 2021.
25. Angelopoulos & Bates. *Conformal Prediction: A Gentle Introduction* / *Conformal Risk Control.*
26. Thirumuruganathan et al. *DeepBlocker: Deep Learning for Blocking in Entity Matching.* VLDB 2021.
27. Konda et al. *Magellan: Toward Building Entity Matching Management Systems.* VLDB 2016.
28. Mudgal et al. *Deep Learning for Entity Matching (DeepMatcher).* SIGMOD 2018.
29. Wu et al. *ZeroER: Entity Resolution using Zero Labeled Examples.* SIGMOD 2020.
30. Papadakis et al. *Blocking and Filtering Techniques for Entity Resolution: A Survey.*
31. Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor Search.* PAMI 2011.
32. Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs (FAISS).*
33. FlexTok; DetailFlow; Spectral Image Tokenizer; CausalEmbed; CoFiRec (autoregressive coarse-to-fine generation family).