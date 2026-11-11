# E3-F literature review

## Purpose

This review identifies scientifically relevant foundations for regional
reliability in RGB-thermal face recognition. It distinguishes uncertainty,
feature consistency, matching confidence, and selective prediction. No claim
of novelty is made.

## Uncertainty in face recognition

### Probabilistic Face Embeddings

Yichun Shi and Anil K. Jain. **Probabilistic Face Embeddings**. *Proceedings
of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019,
pp. 6901-6910. DOI: `10.1109/ICCV.2019.00700`.

The method represents a face embedding as a Gaussian distribution and uses
the variance as feature uncertainty for matching and fusion. It is the
closest principled precedent for uncertainty attached to face features.
Its uncertainty is not automatically a calibrated regional reliability
measure, and the original matching procedure is more expensive than a
cosine score.

Reference: <https://arxiv.org/abs/1904.09658>

### Data Uncertainty Learning in Face Recognition

Jie Chang, Zhonghao Lan, Changmao Cheng, and Yichen Wei. **Data Uncertainty
Learning in Face Recognition**. *Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 5709-5718.
DOI: `10.1109/CVPR42600.2020.00575`.

This work jointly learns face feature means and variances, treating
uncertainty as data uncertainty rather than attaching a post-hoc variance
head to fixed features. It supports learning uncertainty from recognition
supervision, but does not establish that the same uncertainty is calibrated
under RGB-thermal shift or at individual facial regions.

Reference: <https://arxiv.org/abs/2003.11339>

## Cross-spectral face recognition

### Not Afraid of the Dark: NIR-VIS Face Recognition via Cross-spectral
Hallucination and Low-rank Embedding

Jose Lezama, Qiang Qiu, and Guillermo Sapiro. **Not Afraid of the Dark:
NIR-VIS Face Recognition via Cross-spectral Hallucination and Low-rank
Embedding**. arXiv preprint, 2016.

The low-rank embedding encourages shared structure between NIR and visible
features. This is relevant to cross-modal consistency, but a shared
representation can be jointly wrong; consistency is therefore not
equivalent to reliability.

Reference: <https://arxiv.org/abs/1611.06638>

### Matching Thermal to Visible Face Images Using a Semantic-Guided
Generative Adversarial Network

Cunjian Chen and Arun Ross. **Matching Thermal to Visible Face Images Using
a Semantic-Guided Generative Adversarial Network**. *2019 14th IEEE
International Conference on Automatic Face & Gesture Recognition (FG)*,
2019. arXiv version: 2019.

The method uses semantic facial components and identity/perceptual
objectives for thermal-to-visible matching. It motivates region-aware
cross-spectral supervision, but generated-image quality and identity
preservation are separate risks from reliability calibration.

Reference: <https://arxiv.org/abs/1903.00963>

### Beyond the Visible: A Survey on Cross-spectral Face Recognition

David Anghelone, Cunjian Chen, Arun Ross, and Antitza Dantcheva.
**Beyond the Visible: A Survey on Cross-spectral Face Recognition**.
*Neurocomputing*, accepted survey; arXiv version 2022, revised 2024.

This survey maps heterogeneous feature extraction, comparison, datasets, and
open problems in cross-spectral face recognition. It is useful for positioning
the SF-TL54 setting, but it is not evidence that a specific reliability
strategy works on this dataset.

Reference: <https://arxiv.org/abs/2201.04435>

## Calibration and selective prediction

### On Calibration of Modern Neural Networks

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. **On Calibration
of Modern Neural Networks**. *Proceedings of the 34th International
Conference on Machine Learning (ICML)*, PMLR 70, 2017, pp. 1321-1330.

The paper demonstrates that strong predictive accuracy does not imply
calibrated confidence and evaluates temperature scaling. For this project,
the relevant lesson is that any regional confidence must be calibrated on
held-out subject-disjoint data rather than inferred from loss reduction or
variance.

Reference: <https://proceedings.mlr.press/v70/guo17a.html>

### Selective Classification for Deep Neural Networks

Yonatan Geifman and Ran El-Yaniv. **Selective Classification for Deep Neural
Networks**. *Advances in Neural Information Processing Systems (NeurIPS)*,
2017, vol. 30, pp. 4878-4887.

Selective classification formalizes abstention under a target risk. This is
directly relevant if regional reliability is ultimately used to reject or
down-weight uncertain matches. The original work is classification-focused,
so verification-risk adaptation must be evaluated explicitly.

Reference: <https://arxiv.org/abs/1705.08500>

## Region-aware and robustness precedents

### On Improving the Generalization of Face Recognition in the Presence of
Occlusions

Xiang Xu et al. **On Improving the Generalization of Face Recognition in the
Presence of Occlusions**. arXiv technical report, 2020.

The OREO approach uses local identity-related regions for occlusion-aware
recognition. It is a useful region-aware precedent, but it is not a verified
uncertainty-calibration method and is not specific to RGB-thermal matching.

Reference: <https://arxiv.org/abs/2006.06787>

## E3 implications

The literature supports:

1. modeling uncertainty as a distributional or variance-like quantity when
   the training objective identifies it;
2. calibrating confidence against held-out outcomes;
3. evaluating selective risk and coverage;
4. using semantic/local regions for robustness analysis.

It does not support treating:

* ranking gaps as calibrated uncertainty;
* cross-modal agreement as independent evidence of correctness;
* transformation consistency as ground-truth reliability;
* increased uncertainty variance as calibration.

These caveats agree with E3-D4 and E3-E4, where ranking ordering improved but
clean correlation remained weak and stochastic degradation targets were not
sufficiently repeatable or region-specific.
