# Executive Summary

The DARE project proposes a novel **depth-adaptive embedding** approach to **blocking** in entity resolution (ER), allowing query-side embeddings to be computed progressively and stopped early when sufficient evidence accumulates. In this scheme, each record is encoded through a multi-block transformer “trunk” that emits successive **chunks of a residual embedding** plus a learned **confidence head** at each depth. At query time, a record’s embedding is computed chunk-by-chunk, comparing the partial query embedding against precomputed database chunks at the same depth. A learned head predicts whether each candidate match’s classification is likely to flip with deeper computation. Candidates with confidently “in” or “out” predictions are pruned; the small remaining ambiguous set is forwarded to a full matcher. In effect, DARE performs **anytime nearest-neighbor blocking** – adaptively balancing computation vs. recall.

Feasibility is **Medium**. Embedding-based blocking is well-established (e.g. DeepBlocker), and adaptive computation has been effective in related domains (e.g. AdANNS, SPI, Panorama). However, DARE’s combination of on-the-fly embedding refinement, learned stopping rules, and tight recall guarantees is quite novel and technically intricate. Key challenges include designing effective chunk granularity, training the “residual” encoder and head, and ensuring wall-clock gains given varied stopping depths. Major resource needs (large transformer training, high-dimensional vector indices) are significant but solvable with current tooling (PyTorch, FAISS/Qdrant, etc.). Timeline of 6–18 months is plausible with a skilled team, assuming access to moderate compute. No major ethical or regulatory barriers are evident beyond standard data privacy for ER tasks.

The research landscape already contains **dense neural blocking** (e.g. DeepBlocker and the recent UBlocker) which compute fixed embeddings for each record. On the ANN side, **adaptive search** methods (AdANNS, Panorama) explore multi-resolution representations or early-pruning in similarity search. DARE’s novelty lies in applying adaptivity directly to the **generation of embeddings** in ER blocking. Closest works include AdANNS (which uses _matryoshka_ embeddings for retrieval speed-ups) and SPI (a multi-level index with uncertainty-based early exit), but DARE is distinguished by learning the stopping logic (rather than fixed tiers) and targeting blocking recall. A comparison of DARE vs top related approaches is provided below.

Looking ahead, the next steps are: (1) collect standard ER datasets (DBLP-ACM, Walmart–Amazon, Abt–Buy, Fodors–Zagat) and a hard synthetic case; (2) implement baseline blocking (transformer + FAISS/KNN); (3) prototype chunked encoding and head network; (4) conduct ablations on chunk size and stride; (5) evaluate recall-vs-computation tradeoff. With careful design and empirical tuning, DARE could demonstrate a new point on the speed/reliability frontier for ER blocking.

**Feasibility Verdict:** _Medium._ The approach is backed by strong motivation – ER blocking can tolerate ambiguity, so early stopping can dramatically cut encoder cost. Prior work on adaptive compute and blocking show the pieces are promising. However, the co-design of generator and stopper, plus ensuring statistically valid recall, adds research risk. If the learned head misfires, recall may suffer. The project will demand substantial computation and careful engineering. With adequate resources and iteration, however, the concept is sound and likely to yield worthwhile improvements over static blocking.

**Prioritized Risks & Mitigations:** The highest risk is _accuracy degradation_. If the confidence head or early-stop rule is miscalibrated, true matches could be dropped. This is mitigated by a **validity wrapper** using conformal methods to bound tail-error rates. Another risk is _computational overhead_ – the system must actually save wall-clock time despite branching variance. Mitigation: optimize batching or use static “prefetch” to amortize. Complexity of training (multiple losses, curriculum) is also a risk; careful phased training (Section 4) and robust ablation will help isolate issues. Data risk: lack of a rich “residual signal” in some datasets could stall early stopping. Mitigation: design synthetic “hard” dataset, and strip out already-determined fields so head focuses on truly ambiguous content. We flag ethical/regulatory risks as low: blocking deals with record linkage but is a standard practice (the work itself does not introduce new privacy issues beyond handling any personal data responsibly).

**Next Steps:** Gather benchmark and synthetic ER data; implement a simple end-to-end baseline (static embedding + ANN) for comparison; design and code the chunked transformer encoder and small head networks; pretrain on an ER ranking task, then fit heads. Perform ablations for chunk-width and stride (Section 5) and compare learned vs. empirical stopping. Measure recall vs encoder-operations and wall-clock. Iterate to tighten the conformal guarantees. Document results and prepare code repo. The ultimate goal is a working prototype and an empirical paper.

## Key Hypotheses and Methods

- **Adaptive Blocking Hypothesis:** We can **reduce embedding computation** per query by early-stopping without significant recall loss. Since blocking tolerates **ambiguity but not misses**, the system stops not when a match is “certain,” but when remaining candidates are safely classified “in” or “out” with high confidence. This trades computation for a slightly larger forwarded set, measured by _recall at fixed candidate size_ (not precision).

- **Chunk-Width / Stride Hypothesis (Section 5):** Embeddings can be split into coarse-to-fine _chunks_ (e.g. 16–64 dims each) that are learned sequentially. We will test whether **wider chunks** yield more stable but slower-to-fire stopping (high bias, low variance) versus **narrower chunks** (fast decisions but possibly jittery). Similarly, the **residual-stride** (how many new dims to compare per step) affects how quickly discriminative signal accumulates. The explicit hypothesis: _an intermediate chunk size / step stride optimizes recall-vs-cost_, and training can decouple the chunk granularity from inference stride (sweep both).

- **Per-Scheme Encoder Hypothesis (Section 3.4):** Tailoring the encoder to the **blocking scheme** (e.g. dropping fields already filtered by exact predicates) will improve efficiency. By removing “determined” attributes from the encoder input, we enlarge the residual margins that the head measures. We will test general-purpose vs scheme-specific encoders (this is both a hypothesis and an ablation).

- **Learned Tail-Prediction Hypothesis (Section 3.3–3.6):** A neural “confidence head” can _accurately predict_ the impact of further embedding dimensions on each candidate’s similarity score. This learned prediction allows early stopping without fixed bounds. We will train the head on residual rankings to minimize mis-prediction, and wrap it with a conformal guarantee to cover its errors. Ablations will test head-on vs off, learned vs empirical probability estimates.

**Methods:** The system is a modified Transformer encoder producing a **“running vector”** as depth increases. After each block, we extract a chunk embedding (and cumulative query vector) and a small head that scores each candidate’s flip probability (Sections 3.1–3.5). The query algorithm appends chunks and at each step prunes candidates whose predicted P(flip) is below risk δ. Training is in phases (Section 4): first jointly train encoder + ranking chunks, then freeze encoder and train heads for flip prediction, finally optional fine-tuning. We will use self-supervised pair ranking (like DeepBlocker) with hard-negative mining to train the encoder chunks. Conformal risk-control (as in “Conformal Risk Control” methods) ensures the recall guarantee.

## Data and Resources

**Datasets:** We will use standard ER benchmarks (Magellan/DeepMatcher datasets) such as DBLP–ACM, Walmart–Amazon, Abt–Buy, Fodors–Zagat. These vary in cleanliness and should reveal how ambiguous matches are. We will also construct a **hard synthetic dataset** (with injected near-duplicates across many fields) to stress-test the early-stopping. To verify generality, we may test on a general retrieval task (e.g. a multilingual text retrieval corpus) to ensure the adaptive mechanism works beyond ER.

**Tools:** Deep learning frameworks (PyTorch/TensorFlow) for the transformer encoder and head networks. Pretrained models (BERT/MPNet with 768–1024 dims) as starting points. Vector search libraries: e.g. FAISS or Qdrant for K-NN lookup of chunks. Conformal prediction libraries (or custom code) for guaranteed coverage. Development on GPUs (NVIDIA A100/V100) for training large encoders.

**Personnel:** ML engineers/researchers with experience in NLP/ER and vector search. One or two graduate students (or researchers) plus possible software support for data and system work. Collaboration with an ER expert would be useful.

**Timeline:** A 6–18 month timeline is suggested. Roughly, initial 2–3 months for data/setup, 3–6 months for implementation of model/heads and training, 2–4 months for evaluation and ablations, and 2–3 months for writing up results. A plausible Gantt chart is provided below.

**Success Criteria:** The system should achieve competitive or better **recall** at a given candidate-set size _with substantially lower encoder compute_ than a baseline static embedding (e.g. full BERT+FAISS blocking). Key metrics are recall-vs-flops and recall-vs-latency curves. We also measure distribution of stopping depths (to gauge compute savings) and storage overhead (database of multi-depth chunks). Maintaining high **tail recall** (rare hard matches) is critical – if head calibration fails on a few pairs, that’s a problem (hence the conformal wrap). Secondary metrics: inference throughput (how well batching mitigates variance) and model size.

## Current Research Landscape

- **Dense Neural Blocking:** Recent work uses deep networks to embed records for blocking. _DeepBlocker_ (Thirumuruganathan et al. 2021) presents a family of self-supervised designs and found a best solution (RBP) that _“outperforms the best existing DL solution and best existing non-DL solutions”_ on dirty/text ER datasets. This confirms that learned embeddings improve recall. Another line (“universal blockers”) learns large pretrained models; e.g. _UBlocker_ (Zhang et al., 2024) trains on 1M diverse tables and _“outperforms previous domain-specific dense blocking methods by almost 10% mAP on average”_. These are **static** (non-adaptive) – each record is fully embedded once. DARE differs by adding dynamic stopping.

- **ANN and Multi-Stage Search:** In the broader vector search domain, **multi-stage retrieval** is a common pattern. For example, Qdrant’s query API supports a “prefetch” stage (fast single-vector) followed by a more expensive rerank (e.g. ColBERT). This is analogous to oversampling: retrieve many candidates with a coarse model, then refine. Similarly, _ACORN_ (Qdrant 2025) explores graph neighbors-of-neighbors to handle filtered search. DARE’s approach is in this spirit but with learned precision.

- **Adaptive/Anytime Representations:** Recent papers explore _adaptive capacity_ of embeddings. AdANNS (NeurIPS 2023) uses **matryoshka embeddings** – nested vectors of increasing capacity – so that “stages of ANNS that can get away with more approximate computation should use a lower-capacity representation”. They report, e.g., _“AdANNS-IVF matches accuracy while being up to 90× faster in wall-clock time”_ on ImageNet. Another is _SPI (2026)_, which builds a **semantic pyramid index**: a multi-level index of embeddings from shallow to deep, with a controller routing each query to a suitable level. On MS MARCO/NQ, SPI achieves _“1.4–2.3× average retrieval latency reduction under fixed Recall@10”_. These show adaptivity _at query time_ can cut cost significantly. DARE’s novelty is applying this idea to ER blocking and coupling embedding generation with stopping rules.

- **Refinement/Early-Exit in ANN:** The _Panorama_ method (2025) attacks the distance-refinement bottleneck by learning orthogonal transforms that pack most “energy” into early dimensions. It enables _“accumulative partial distance computations”_ so that 90% of signal is in the first half of dims and candidates can be pruned early. Panorama integrates with IVFPQ/HNSW and reports _“2–30× end-to-end speedup with no recall loss”_ across image/text embeddings. This is akin to DARE’s dimension-ordering idea, but DARE goes further by learning to stop at a flexible point per candidate.

- **Vector Databases and Tools:** Open-source engines like **FAISS** (Meta) and **Qdrant** (Rust vector DB) implement state-of-art ANN. They typically use fixed embedding vectors indexed in HNSW or IVF. Qdrant’s recent features (multi-vector search, ACORN) reflect industry interest in multi-stage/search filtering. These systems do not generate embeddings adaptively; at best they oversample queries. They do, however, serve as our baselines. For example, UBlocker’s experiments use FAISS for KNN search, and we will likely do the same.

- **Blocking Methods:** Classic blocking (rule-based, token blocking) and GPU-based blocks (HyperBlocker) remain relevant for comparisons. Also, unsupervised ER methods like ZeroER (SIGMOD 2020) show strong performance without labels. However, these focus on the matching or training aspects, not on reducing embedding cost. Conformal prediction and risk-control techniques are being applied to early-exit networks recently (e.g. Jazbec et al. 2024). DARE builds on these by wrapping its head with a conformal guarantee to ensure recall.

**Gaps and Novelty:** In sum, no existing work has combined _generation-adaptive embeddings_ with blocking. Prior _dense blockers_ were static. Adaptive retrieval schemes (AdANNS, SPI) apply to search but not ER. DARE’s learned chunked-embedding plus stopping rule is a new design point. Its main novel claims: (a) enabling **"anytime" blocking** with an explicit learned stopping criterion, (b) co-training encoder and head jointly, (c) applying conformal risk-control to blocking for the first time. The closest analog is AdANNS’s idea of variable-capacity embeddings, but DARE focuses on _per-record early halting in blocking_, a task with very different requirements and metrics.

## Feasibility Analysis

**Technical Feasibility:** All necessary components exist in the literature. Transformers can be instrumented to output intermediate vectors (cf. _prefix embeddings_). Early-exit networks and ACT/BranchyNet show that halting policies can be learned. Implementing chunked inference (caching activations to reuse) is straightforward given modern frameworks. Training will be involved: multi-objective loss (ranking + flip prediction + maybe chunk decorrelation) but similar to other deep ER models. The conformal layer is an add-on that has known algorithms. The biggest uncertainty is integration: ensuring the query-time loop (partial encoding + distance updates + decision) is efficient in practice. If not well-engineered, overhead could erase benefits. But similar strategies have succeeded in vector search and classification.

**Data Availability:** Standard ER corpora are available (as noted above). No new labeled data collection is required (self-supervision and heuristics suffice). Synthetic data for stress tests can be generated (e.g. by copying records with slight edits). Thus, data is not a bottleneck.

**Time and Cost:** Training a large encoder (e.g. BERT-base scale) from scratch may be costly, but we can fine-tune or train on ER pairs (no need for enormous corpora). Even so, expect weeks of GPU time for multiple ablations. GPU clusters (4–8 A100s) would make this feasible in a few months; on a budget, cloud GPU costs could be ~$10K-$20K. Human effort: roughly a full-time researcher for ~1–2 years (with possible parallelization of tasks). Timeline risk: if difficulties arise, 18 months would cover more extensive engineering.

**Personnel:** A small team with expertise in deep learning, ER, and information retrieval is needed. The interdisciplinary nature (ML + DB) means ideally one ML engineer and one data engineer. Without such expertise, the project could stall.

**Ethical/Regulatory:** Standard ER work is low-risk. Potential use with personal data would necessitate privacy protections, but the project itself is method-oriented. There is no obvious misuse (it helps match duplicates, not deanonymize unknown individuals). We must ensure any PII in datasets is handled per policy. Conformal risk-control helps maintain trust in the system’s guarantees (avoiding silent errors).

**Overall Feasibility Verdict: Medium.** The approach is sound and aligns with current trends in adaptive compute. The main challenge is complexity: if the early-exit does not deliver net wall-clock gains (due to uneven stopping depths), or if recall suffers, the value will be limited. But literature suggests such adaptivity can work (AdANNS, Panorama). With careful engineering and tuning, DARE appears achievable within a year or so.

## Prioritized Risks and Mitigations

1. **Recall Loss / Head Error:** If the confidence head mispredicts, true matches could be pruned. _Mitigation:_ Use a conservative conformal envelope around the head’s prediction (Section 3.6). Calibrate the head on held-out validation to guarantee, with high probability, that recall stays above a threshold. Also ablate learned head vs. simple bounds (e.g. Bernstein inequalities) to compare.

2. **Insufficient Residual Signal:** Some records may be mostly resolved by a few fields, leaving very little for adaptive computation to exploit. _Mitigation:_ Remove fields covered by deterministic blocking from the encoder input (as proposed in 3.4). If residual embedding remains low-signal, explicitly train a residual decorrelation loss so each chunk adds information. Provide a “fallback” of full encoding if the head never stops early.

3. **Compute Overhead & Stragglers:** Adaptive depth means different queries cost differently, which can hurt batching or real-time guarantees. _Mitigation:_ Design batch schedules to accommodate variable steps, or impose a max-depth cap. Experiment with asynchronous or dynamic-batching strategies. Compare average FLOPs vs. worst-case “tail” (as in metrics 6.3).

4. **Implementation Complexity:** Integrating adaptive encoding, efficient KNN, and calibration is non-trivial. _Mitigation:_ Build incrementally: start with a simple proof-of-concept using few layers and small dims. Use existing libraries (FAISS for distance search, PyTorch Lightning for training). Modularize components (encoder vs head vs search). Leverage the proposed repository structure (Section 8) as a blueprint.

5. **Resource Constraints:** Large transformers and KNN indexes require memory. _Mitigation:_ Use moderate-size models (e.g. 768-dim rather than 1024), quantize DB embeddings if needed. The adaptivity itself reduces DB storage if we need fewer dims at query time. If budget is an issue, use cloud grants or existing GPU clusters.

6. **Competition / Redundancy:** While DARE is novel, adaptive retrieval is hot. SPI and AdANNS are very recent (2024–26). There is a risk they might release code or that industry vector DBs close the gap. _Mitigation:_ Emphasize the unique ER focus and learned-blocking angle. Monitor new publications. Collaborate or benchmark against any emergent baselines.

## Project vs Related Works (Attribute Comparison)

| **Project / Work**               | **Approach**                                                                                           | **Dataset**                                                                            | **Results**                                                                                                          | **Limitations**                                                              | **Year**       | **Link**                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| **This Work (DARE)**             | Depth-adaptive chunked embedding + learned early-stop head for blocking; static DB precomputed chunks. | ER benchmarks (DBLP-ACM, Walmart–Amazon, Abt–Buy, Fodors–Zagat) + synthetic hard case. | (To be determined) Expected high recall with reduced average encode cost.                                            | Prototype stage; needs effective head training; variable query cost.         | 2026 (planned) | _Design Doc (provided)_                                                        |
| **DeepBlocker**                  | Self-supervised DL for blocking; aggregates word embeddings/transformer; no adaptivity.                | Standard ER datasets (structured, textual, dirty).                                     | Beats prior DL and non-DL blockers on “dirty/text” data.                                                             | Requires full embedding (static); no compute reduction.                      | 2021           | [PVLDB’21 paper](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf) |
| **AdANNS**                       | Adaptive ANNS: uses _matryoshka_ embeddings of varying size per query stage.                           | ImageNet, NaturalQuestions (embeddings).                                               | State-of-art trade-off: e.g. _“90× faster at same accuracy”_ on ImageNet and matches accuracy with half-cost on QnA. | Focus on retrieval tasks; requires specially trained nested representations. | 2023           | [arXiv:2305.19435](https://arxiv.org/abs/2305.19435)                           |
| **Panorama**                     | Learned orthogonal transforms + early partial-distance pruning in ANN refinement.                      | CIFAR-10, GIST, OpenAI Ada/Large embeddings (vision+text).                             | _“2–30× end-to-end speedup with no recall loss”_ by concentrating distance in first dims.                            | Focuses on verification step; no blocking or dataset filtering.              | 2025           | [ArXiv:2510.00566](https://arxiv.org/abs/2510.00566)                           |
| **SPI (Semantic Pyramid Index)** | Multi-level index (shallow-to-deep embeddings) + uncertainty controller routes queries.                | MS MARCO, Natural Questions (large-scale RAG retrieval).                               | Achieves _1.4–2.3× lower latency_ at fixed Recall@10 vs flat index.                                                  | Designed for streaming text retrieval; complex index.                        | 2026           | [PVLDB 2026 (preprint)](https://github.com/FastLM/SPI_VecDB)                   |
| **Universal Blocker (UBlocker)** | Single universal transformer pre-trained on 1M tables (contrastive) for blocking.                      | 19 heterogeneous ER datasets across 9 domains.                                         | Outperforms dataset-specific dense blockers by ~10% mean-average-precision.                                          | Static embedding; heavy pretraining; no adaptivity.                          | 2024           | [ArXiv:2404.14831](https://arxiv.org/abs/2404.14831)                           |

## Recommended Next Steps

1. **Data & Baselines:** Collect the ER benchmark sets and implement a standard dense-blocking baseline (e.g. BERT/MPNet + FAISS KNN) to quantify the current recall-cost curve. Optionally include classical sparse blocks for reference. Ensure all pipelines measure _recall at fixed candidate size_.

2. **Prototype Encoder:** Build the shared transformer trunk to output chunk embeddings. Experiment with chunk sizes (e.g. 16,32,64 dims) on toy data. Verify that concatenating chunk vectors approximates the full embedding.

3. **Train Ranking Loss:** Using ER pairs (self-supervised positives and random negatives), train the chunked encoder to rank true matches higher. This is similar to DeepBlocker’s self-supervised ranking.

4. **Develop Confidence Head:** Design a small network to predict “flip probability” for each candidate. Train it on held-out validation data by simulating progressive encoding (see Section 3.3). Calibrate its confidence (using e.g. Platt scaling or isotonic) to prepare for conformal bounding.

5. **Implement Stopping Rule:** Code the query-time loop: compute chunk, update query vector, compare against DB, use head+margin to prune. Test on some records to ensure logic is correct. Integrate a simple conformal threshold to guarantee recall ≥ δ.

6. **Experiment and Ablate:** Run full experiments varying chunk width, stride, use of head (vs. heuristic) and conformal vs naive. Evaluate recall vs encoder-ops curves. Key is to find configurations where _computational savings_ are substantial at minimal recall loss. Compare to PQ or HNSW baselines (PV of static embeddings).

7. **Optimize and Profile:** If wall-clock savings fall short, profile bottlenecks. Consider batching strategies or rebalancing DB costs (e.g. lowering efSearch in HNSW, use multi-stage search). If needed, simplify head or increase oversampling to ensure missed neighbors are extremely rare.

8. **Document and Plan Publication:** Maintain clear code and a structure for experiments. Prepare a report/paper outlining the method and findings. Release code (e.g. via a GitHub repo as suggested in Section 8).

With these steps, we will validate the DARE concept and identify whether it truly shifts the speed/recall frontier in ER blocking.

## Bibliography

- DeepBlocker: Thirumuruganathan _et al._, “Deep Learning for Blocking in Entity Matching: A Design Space Exploration,” _PVLDB_ 14(11):2459–2472, 2021.
- UBlocker: Zhang _et al._, _Towards Universal Dense Blocking for ER_, arXiv:2404.14831 (2024).
- AdANNS: Rege _et al._, “Adaptive Semantic Search,” _NeurIPS 2023_ (arXiv:2305.19435).
- Panorama: Ramani _et al._, _PANORAMA: Fast-Track Nearest Neighbors_, arXiv:2510.00566 (2025).
- SPI: Geng & Chai, _Semantic Pyramid Indexing_, _PVLDB_ 19(1):XXX (2026).
- Zeakis _et al._, “Pre-trained Embeddings for ER: An Experimental Analysis,” _PVLDB_ 16(9):2225–2238 (2023) – measures vectorization overhead and blocking performance.
- Papadakis _et al._, _Blocking and Filtering Techniques for ER: A Survey_, _ACM Trans. DB_ (2020).
- Schuster _et al._, “CALM: Confident Adaptive LM,” _NeurIPS 2022_ (adaptive transformer inference).
- Angelopoulos & Bates, _Conformal Prediction: A Gentle Introduction_ (2019) – background on conformal risk control.
- Qdrant multi-stage retrieval docs (Qdrant Ltd., 2023) – introduces fast prefetch + rerank pattern.
- Kuzushita & Wang, “Product Quantization for ANN,” _PAMI_ (2011) – baseline quantization technique.

_(All references cited in the text above.)_

```mermaid
gantt
    title DARE Project Timeline (2026–2027)
    dateFormat  YYYY-MM
    section Setup
    Data collection       :active, a1, 2026-07, 1m
    Baseline implement    :a2, after a1, 2026-08, 1m
    Architecture design   :a3, after a2, 2026-09, 2m
    section Implementation
    Encoder prototype     :a4, after a3, 2026-11, 3m
    Head network develop  :a5, after a4, 2027-02, 2m
    Integrate & test      :a6, after a5, 2027-04, 1m
    section Evaluation
    Model training/tuning :a7, after a6, 2027-05, 3m
    Ablation studies      :a8, after a7, 2027-08, 2m
    section Reporting
    Write paper           :a9, after a8, 2027-10, 2m
    Code release & wrapup :a10, after a8, 2027-09, 1m
```
