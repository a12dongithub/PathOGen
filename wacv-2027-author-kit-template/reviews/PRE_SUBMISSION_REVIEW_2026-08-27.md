# Pre-Submission Review: CPathOGen

**Target venue:** WACV 2027  
**Primary scope:** Sections 1 (Introduction) and 2 (Related Work)  
**Review date:** 2026-08-27  
**Method:** Eight independent review perspectives, consolidated without double-counting repeated findings  
**Manuscript changes:** None

## Executive verdict

**Current recommendation: Weak Reject in the Algorithms Track; promising after major revision, with substantially stronger fit to the Evaluation & Dataset Track.**

The paper has a valuable core idea: use a controllable pathology generator as a measured perturbation engine, then audit downstream models under named changes to cellular organization, nuclear morphology, and appearance. Same-noise dose sweeps, a random-real-tile negative control, and out-of-loop analyzers make this more rigorous than a purely visual counterfactual demonstration.

The present Introduction and Related Work nevertheless promise more than the body currently establishes. The largest problems are an unclear Algorithms-track novelty claim, a ControlNet/method mismatch, incomplete downstream definitions and protocol, and use of “verified,” “preserved,” and “controlled” without documented per-pair acceptance criteria. These are submission-level issues, not copy-editing issues.

## WACV 2027 fit and observable compliance

The official [WACV 2027 Author Guidelines](https://wacv.thecvf.com/Conferences/2027/AuthorGuides) allow eight content pages, with additional pages containing references only, and set a 50 MB PDF limit. The [Reviewer Guidelines](https://wacv.thecvf.com/Conferences/2027/ReviewerGuidelines) describe the Algorithms Track as requiring algorithmic novelty and quantified evaluation against current alternatives. The Evaluation & Dataset Track explicitly covers tools, datasets, benchmarks, and practices for testing, stress-testing, auditing, comparing, and interpreting AI systems.

The delivered PDF passes the visible page, size, template, anonymity, font, and rendering checks: technical content ends on page 8; pages 9–10 contain references; the file is about 22.1 MB; Times-compatible regular, bold, and italic fonts are embedded; and no clipped figures, broken tables, missing images, or unresolved citation markers are visible.

Two upload blockers remain:

- `main.tex` still uses paper ID `*****`, which appears on every page.
- Page 8 contains an unfinished acknowledgment, “This work was made possible by the xxx.” The anonymous review version should not contain this placeholder or identifying acknowledgments.

## Consolidated severity summary

- **Critical:** 6
- **Major:** 13
- **Minor/polish:** 12

## Critical findings

### C1. The current track and contribution framing do not align

The source selects the Algorithms Track, but Sections 1–2 do not identify a clearly new algorithmic mechanism or provide quantified comparison with a current alternative generator. ControlNet, FiLM, latent diffusion, shared-noise sampling, and candidate selection are presented as a useful combination, but the technical delta is not isolated.

The strongest framing is instead: **a fidelity-audited counterfactual framework for controlled pathology-model evaluation**. This maps almost exactly to the Evaluation & Dataset Track. Merely switching tracks is not sufficient—the benchmark/protocol, dataset artifact, downstream evaluation, and release plan must still be fully specified—but it makes the central contribution easier to defend.

Evidence: `main.tex:5`; `sec/1_introduction.tex:25–29`.

### C2. “ControlNet” conflicts with the architecture described in Method

The abstract and Related Work state that cellular maps condition synthesis through ControlNet. Method instead describes a convolutional map encoder, channel/latent concatenation, and expansion of the first U-Net convolution from four to eight channels. It does not describe the duplicated conditioning branch and zero-convolution connections normally associated with ControlNet.

This affects reproducibility and the first contribution. Either the Method omits the actual ControlNet branch, or Sections 1–2 use an inaccurate architecture label.

Evidence: `sec/2_related_work.tex:22`; `sec/3_method.tex:8–18`.

### C3. “Verified counterfactuals” and preservation claims exceed the evidence

The Introduction says each requested change is verified while other factors are preserved, and Related Work says downstream pairs pass generation-side checks. The manuscript reports strong aggregate correlations and dose-response behavior, but not a complete per-pair protocol: acceptance thresholds, non-target tolerances, rejection/acceptance rates by intervention, retained downstream counts, and the pair manifest are absent.

Holding non-target conditioning values and the noise seed fixed is not evidence that all non-target visual properties are preserved. The defensible present claim is aggregate intervention fidelity with modeled non-target controls held fixed.

Evidence: `sec/1_introduction.tex:21,27–29`; `sec/2_related_work.tex:28`; `sec/3_method.tex:42–46`; `sec/5_results.tex:229`; `sec/7_discussion.tex:8`.

### C4. The downstream contribution is not presently reviewable

The Introduction claims completed analysis of classifiers, frozen foundation encoders, and survival models. Section 6 still says its populated table “reserves” results and does not define TVD, prediction-flip rate, or BNR operationally. It also lacks downstream sample sizes, splits, head-training details, aggregation, uncertainty, and substantive result interpretation. Explanation-faithfulness analysis is promised but not reported in the stated top/random/bottom form.

The Introduction should not make a completed-study claim until this protocol and its findings are finalized. If BNR remains a claimed contribution, define it formally and include the key finding in the Introduction.

Evidence: `sec/1_introduction.tex:23–29`; `sec/6_probing.tex:4,27`.

### C5. Numerical contradictions undermine trust in the promises made by Sections 1–2

- Experiments says FID/KID use 2,000 generated images; Results says 10,000.
- Experiments defines five doses as `{-1,-0.5,0,0.5,1}` SD; Results labels them `{-2,-1,0,1,2}` SD.
- The Real-vs-Real KID matches the unfiltered-generator KID exactly, while a source comment says to recalculate it.

Although these occur later, they determine whether the Introduction’s evaluation claims are supportable.

Evidence: `sec/4_experiments.tex:9,19`; `sec/5_results.tex:10,17–19,161`.

### C6. The review PDF is not upload-ready

Replace the `*****` paper ID and remove the unfinished acknowledgment before submission. WACV warns that anonymity/template violations can lead to rejection.

Evidence: `main.tex:16`; `sec/8_acknowledgement.tex:1–4`; PDF pages 1–8.

## Major findings

1. **Closest literature is incomplete.** Related Work should compare CPathOGen directly with [Yang et al., MIDL 2026](https://proceedings.mlr.press/v315/yang26a.html), [TopoCellGen, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_TopoCellGen_Generating_Histopathology_Cell_Topology_with_a_Diffusion_Model_CVPR_2025_paper.html), [PathDiff, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bhosale_PathDiff_Histopathology_Image_Synthesis_with_Unpaired_Text_and_Mask_Conditions_ICCV_2025_paper.html), and Morphology-Focused Diffusion (WACV 2023). [Histopath-C, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Noori_Histopath-C_Towards_Realistic_Domain_Shifts_for_Histopathology_Vision-Language_Adaptation_WACV_2026_paper.html) is directly relevant to controlled pathology-model robustness. ECED (WACV 2025) is also relevant to the generic counterfactual paragraph.

2. **The MoPaDi distinction is not precise enough.** A defensible contrast is prediction-targeted counterfactual explanation versus feature-specified model auditing. The present statement that correlated attributes may change is an unsupported criticism unless shown or cited. Keep the MoPaDi citation attached to its factual description.

3. **Introduction and Related Work repeat the same material.** Generic counterfactual methods, pathology foundation encoders, and CellViT++/HoVer-Net/StarDist roles are summarized twice. Related Work should synthesize differences; the Introduction should motivate the gap and state the contribution.

4. **Related Work contains the paper’s own experimental protocol.** Its final realism/fidelity paragraphs describe what “we report” and which pairs enter downstream probing. Move protocol details to Method/Experiments and use the recovered space for comparison with closest work.

5. **The contribution bullets do not state a memorable result.** They list a generator, datasets, and an analysis, but do not say what the analysis found. The abstract emphasizes BNR while the Introduction omits it. The bullets should identify the precise artifact and one or two completed findings, without implementation-level detail.

6. **The dataset contribution is ambiguous.** “Counterfactual datasets” may mean one benchmark with several intervention families. Define names, sizes, source cohorts, splits, acceptance rates, and release status. WACV states that a dataset claimed as a major contribution must be ready by camera-ready.

7. **Training-data provenance is insufficient.** “1.4 million tiles from 1,114 WSIs” needs direct cohort/data-release attribution and support elsewhere in the paper. “One patient per WSI” should be replaced by the intended relation, likely “1,114 WSIs from 1,114 patients.”

8. **The motivated and evaluated model panels do not match.** Sections 1–2 name CTransPath, UNI, Virchow, CONCH, and Prov-GigaPath; the downstream table uses CTransPath, UNI2-h, and PathLUPI+CONCH, with Virchow and Prov-GigaPath absent. PathLUPI is undefined/uncited, and the original UNI citation does not automatically document UNI2-h.

9. **Cross-dataset generalization is not tested.** “Across image sets” suggests external hospitals, scanners, organs, or datasets. The reported work tests sensitivity to controlled synthetic feature variation within one broad H&E domain. Separate those questions or add external-domain evaluation.

10. **Several claims need narrower evidence boundaries.** “Biologically plausible,” “realistic tissue structure,” “independent measurements,” and “these mechanisms support separate interventions” overstate FID/KID, analyzer agreement, or architectural separation. Prefer “feature-distribution agreement,” “out-of-loop analyzers,” and “separate conditioning channels” unless pathologist review or non-target-drift evidence is added.

11. **The site-confounding sentence is only partly supported.** Howard et al. supports persistent site signatures, but not necessarily every named association among nuclear shape, cell density, tumor–immune organization, staining, and institution. Add feature-specific support or narrow the sentence.

12. **Figure 2 placement and caption need work.** The figure is introduced in Related Work but floats to PDF page 4 after Method has begun. Its caption is not self-contained, and some internal labels are small.

13. **The successful build provenance is missing.** The live `main.log` is from an earlier failed build even though the delivered PDF is complete. Preserve a fresh successful Tectonic log and intermediates for final QA.

## Citation and prose audit

Checks that pass:

- All 34 citation keys used in Sections 1–2 exist in `main.bib`.
- Named models in these sections, including CellViT++, are cited at first use.
- Numeric citations use `~\cite{...}`, appear before punctuation, and render in WACV style.
- No manually typed citation numbers or unresolved citation markers were found.

Items to correct or polish:

- Attach citations to individual foundation-model names rather than one dense cluster; the current citation-key order does not mirror the model order.
- Verify the Tinaz et al. bibliography metadata; its page range and volume field appear inconsistent with the final proceedings record.
- Expand hematoxylin and eosin (H&E), The Cancer Genome Atlas (TCGA), feature-wise linear modulation (FiLM), Fréchet Inception Distance (FID), and Kernel Inception Distance (KID) at first use.
- Replace or define “post-hoc”; here it means an explanation produced after model training/prediction rather than built into the model. “After-the-fact explanation” is clearer.
- Replace “input instrument” with “controlled perturbation generator.”
- Do not say cell-analysis models “disagree in taxonomy” when StarDist does not provide a taxonomy; say they differ in outputs/assumptions and that agreement is measured later.
- Use singular “a pathology study” if only one citation supports the sentence.
- Shorten subsection 2.1 so “probing” does not break as `prob-` / `ing` on PDF page 2.
- Make Figure 1’s caption explain the controls, measurement, and audit loop; the current illustration is legible but generic.
- Reduce nonparallel and overloaded sentences, especially the five-model opening cluster and the long analyzer/evaluation sentence.

## Strongest acceptance case

CPathOGen’s defensible novelty is not ControlNet or FiLM in isolation. It is the use of a multi-axis generator as a controlled perturbation engine, with matched-noise interventions, generation-side measurement, out-of-loop analyzers, and downstream model comparison. The reported within-source dose-response results are strong in aggregate, and the random-real-tile control helps show that the measurement agreement is not automatic.

A concise positioning sentence for future revision could be: “CPathOGen turns controllable H&E synthesis into a measured model-auditing protocol, enabling pathology models to be compared under explicit changes to cellular organization, nuclear morphology, and appearance.” This is a review finding, not a requested manuscript edit.

## Prioritized revision order

1. Decide the target track and contribution identity. For Algorithms, isolate a genuinely new mechanism and add current-generator baselines/ablations; otherwise frame the completed work for Evaluation & Dataset.
2. Reconcile the ControlNet architecture claim with the Method and implementation.
3. Complete the downstream protocol and metric definitions; state concrete findings in the Introduction only after they are supported.
4. Document per-pair verification, non-target tolerances, acceptance rates, retained counts, and uncertainty—or narrow “verified/preserved” claims.
5. Rebuild Related Work around direct comparisons to MoPaDi, Yang et al., Spatial Diffusion, TopoCellGen, PathDiff, Histopath-C, and the closest WACV counterfactual work; remove duplicated protocol text.
6. Reconcile the numerical contradictions and dataset/model provenance.
7. Fix submission placeholders and final layout/caption issues.

## Final assessment

The manuscript is **not ready for WACV submission in its current Algorithms-track form**, but the underlying study has a credible and potentially valuable Evaluation & Dataset contribution. The most important next step is not sentence-level polishing; it is aligning the claims, architecture, downstream evidence, and track around the work that has actually been completed.
