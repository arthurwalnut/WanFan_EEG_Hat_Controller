# Four-frequency FBCCA baseline

For setup and operation, see the [workspace README](../../../../README.md) and
[SSVEP guide](../../../../EEG_Realtime/docs/SSVEP_GUIDE.md).

This package is independent of MI and c-VEP. It uses 8/10/12/14 Hz analytical
sine/cosine references (five harmonics) and O2/O1/P8/P7 from the eight-channel,
250 Hz Hat input. The classifier has no supervised fitting stage. A saved
model contains the fixed processing contract, actual SOS coefficients,
rejection thresholds, provenance, and an evaluation report, not learned
templates or neural-network weights.

Chen et al. (2015), *Filter bank canonical correlation analysis for implementing
a high-speed SSVEP-based brain–computer interface*, JNE 12 046008,
[DOI](https://doi.org/10.1088/1741-2560/12/4/046008), equations 3–7, supplies
the CCA reference construction and weighted squared-correlation rule.
Five nested Chebyshev-I banks use passbands 6/14/22/30/38–90 Hz and stopbands
4/10/16/24/32–100 Hz, with weight `n**(-1.25)+.25`; the band-edge and order
recipe is also visible in
[Nakanishi's research implementation](https://github.com/mnakanishi/TRCA-SSVEP/blob/master/src/filterbank.m).
The original paper used zero-phase filtering. This version explicitly replaces
that with continuous causal SOS filtering, including a 50 Hz Q30 notch,
and uses the same streaming state transitions offline and online. It is a local
four-channel, four-target adaptation, not a reproduction of the paper's
40-target setup or reported accuracy. The paper's equation 6 squares each
correlation; this version follows that equation even though the public
`test_fbcca.m` example aggregates unsquared correlations.

`ContinuousPreprocessor.process(raw)` returns `(N,5,8)`. Call `reset()` after
a source switch, corrupt segment or sample gap. `predict_epoch` receives
`(375,5,8)` filtered values, 375 timestamps, and corresponding raw `(375,8)`
values. The caller retains only windows fully inside a loader-approved task
after the first 0.5 seconds (an explicit engineering warmup), every 25 samples.
It must also enforce display visibility, trial boundaries and sample freshness;
the decoder cannot infer those conditions from an epoch alone.

`build_dataset` accepts loader session dictionaries containing metadata,
samples, times, indices, segments and validated trials. Each trial uses
`trial_id,target,onset_s,tail_s,valid,reasons`. Label targets are 0–3 in frequency
order. Each recording requires a unique metadata `id`; duplicate trial IDs
across different recordings are independent. No display name is used as a
label or grouping key. MI/c-VEP contracts, mixed sources, or different named
subjects are rejected.

`train_model` is named for application compatibility: it calibrates only
rejection thresholds and evaluates the fixed FBCCA decoder. At least six valid
whole trials per class are required. For each class, the last `floor(N/3)`
trials form the test split, and all earlier trials calibrate thresholds; with
six trials this is four calibration plus two test trials. There is no
classifier training split. Sorting is stable by recording creation time and
identity, then onset/trial ID. All overlapping windows retain the parent
trial's split. A session can appear in both splits, so the report does not
establish cross-session generalization.

Scores are `sum(w*rho**2)/sum(w)`, bounded 0–1, and **are not probabilities**.
The maximum score and top-minus-second margin determine candidate and rejection.
Thresholds use one median per correctly decoded calibration trial, with fixed
floors. Insufficient correct target coverage selects an explicitly reported
heuristic fallback. Test scores never fit thresholds, filters or frequency
weights. An accepted output has not been validated as an intention detector;
no-attention/rest rejection and physical display timing remain unvalidated.

Only JSON is used for model serialization. `validate_model` checks source,
the complete contract, SOS coefficients, thresholds and non-overlapping
calibration/test identities. `save_model` and `load_model` do not use pickle.
