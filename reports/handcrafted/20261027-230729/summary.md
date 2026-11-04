# Hand-crafted features, 20261027-230729

60 s from the middle of each song; features from a linear STFT at the file's own rate.

Features: fc_hz, notch_db, slope_break, residue_db, sfm_mean, sfm_var, flux_p2m, crest_db, mod_peak_db, mod_codec_db, ripple_db

# training_data, raw

## training_data raw: 427 clips, 117 AI / 310 human, rates [44100, 48000]

| feature | AUC alone | human median | AI median | NaN |
|---|---|---|---|---|
| fc_hz | 0.276 | 1.68e+04 | 1.57e+04 | 0% |
| notch_db | 0.069 | 0.114 | -22 | 0% |
| slope_break | 0.276 | -13.1 | -69.1 | 0% |
| residue_db | 0.144 | -48.8 | -90.9 | 7% |
| sfm_mean | 0.325 | 0.242 | 0.196 | 0% |
| sfm_var | 0.295 | 0.0128 | 0.0103 | 0% |
| flux_p2m | 0.298 | 6.48 | 4.2 | 0% |
| crest_db | 0.156 | 22.2 | 18.6 | 0% |
| mod_peak_db | 0.451 | 4.26 | 3.62 | 0% |
| mod_codec_db | 0.433 | 2.4 | 2.15 | 0% |
| ripple_db | 0.999 | 19.1 | 28.7 | 0% |

| feature set | leave-one-AI-artist-out AUC |
|---|---|
| all | 0.998 |
| bandwidth | 0.982 |
| smear | 0.876 |
| codec | 1.000 |

# training_data, aac128

## training_data aac128: 427 clips, 117 AI / 310 human, rates [44100, 48000]

| feature | AUC alone | human median | AI median | NaN |
|---|---|---|---|---|
| fc_hz | 0.300 | 1.64e+04 | 1.57e+04 | 0% |
| notch_db | 0.078 | -6.05 | -20.5 | 0% |
| slope_break | 0.924 | -212 | -72.8 | 0% |
| residue_db | 0.246 | -62.6 | -84.8 | 0% |
| sfm_mean | 0.422 | 0.206 | 0.189 | 0% |
| sfm_var | 0.248 | 0.0162 | 0.0113 | 0% |
| flux_p2m | 0.257 | 9.34 | 5.59 | 0% |
| crest_db | 0.156 | 22.2 | 18.6 | 0% |
| mod_peak_db | 0.456 | 4.19 | 3.63 | 0% |
| mod_codec_db | 0.417 | 2.41 | 2.05 | 0% |
| ripple_db | 0.979 | 24.2 | 28.9 | 0% |

| feature set | leave-one-AI-artist-out AUC |
|---|---|
| all | 0.994 |
| bandwidth | 0.980 |
| smear | 0.885 |
| codec | 0.979 |

## Clean-fitted model on the re-encoded files (held-out artists)
- all: 0.952
- bandwidth: 0.939
- smear: 0.894
- codec: 0.960

## playlist 20261019-164821-h4-a4-s7 (excerpts of training songs): 8 clips, 4 AI / 4 human, rates [44100]

| feature | AUC alone | human median | AI median | NaN |
|---|---|---|---|---|
| fc_hz | 0.000 | 1.93e+04 | 1.57e+04 | 0% |
| notch_db | 0.000 | 0.472 | -22.3 | 0% |
| slope_break | 0.000 | -14.6 | -89 | 0% |
| residue_db | 0.250 | -53.8 | -86.3 | 0% |
| sfm_mean | 0.312 | 0.25 | 0.197 | 0% |
| sfm_var | 0.000 | 0.0177 | 0.0129 | 0% |
| flux_p2m | 0.250 | 10.3 | 3.8 | 0% |
| crest_db | 0.125 | 21.3 | 18.4 | 0% |
| mod_peak_db | 0.188 | 9.02 | 3.95 | 0% |
| mod_codec_db | 0.188 | 3.39 | 2.89 | 0% |
| ripple_db | 1.000 | 17.1 | 28.5 | 0% |

| feature set | AUC, model fitted on clean files | AUC, fitted on re-encoded files |
|---|---|---|
| all | 1.000 | 1.000 |
| bandwidth | 1.000 | 0.812 |
| smear | 1.000 | 0.938 |
| codec | 1.000 | 1.000 |

## playlist 20261020-061419-h30-a30-s1 (excerpts of training songs): 60 clips, 30 AI / 30 human, rates [44100]

| feature | AUC alone | human median | AI median | NaN |
|---|---|---|---|---|
| fc_hz | 0.271 | 1.78e+04 | 1.57e+04 | 0% |
| notch_db | 0.000 | 0.268 | -21.6 | 0% |
| slope_break | 0.043 | -11.4 | -79.1 | 0% |
| residue_db | 0.288 | -47.7 | -86.2 | 0% |
| sfm_mean | 0.309 | 0.254 | 0.177 | 0% |
| sfm_var | 0.327 | 0.0138 | 0.0119 | 0% |
| flux_p2m | 0.361 | 4.76 | 3.78 | 0% |
| crest_db | 0.201 | 20.8 | 18.2 | 0% |
| mod_peak_db | 0.347 | 5.11 | 3.53 | 0% |
| mod_codec_db | 0.318 | 3.72 | 2.22 | 0% |
| ripple_db | 1.000 | 17 | 28.1 | 0% |

| feature set | AUC, model fitted on clean files | AUC, fitted on re-encoded files |
|---|---|---|
| all | 1.000 | 1.000 |
| bandwidth | 1.000 | 0.853 |
| smear | 0.836 | 0.826 |
| codec | 1.000 | 1.000 |

## phone captures (all): 758 clips, 258 AI / 500 human, rates [32000]

| feature | AUC alone | human median | AI median | NaN |
|---|---|---|---|---|
| fc_hz | 0.585 | 1.6e+04 | 1.6e+04 | 0% |
| notch_db | nan | nan | nan | 100% |
| slope_break | nan | nan | nan | 100% |
| residue_db | 0.417 | -100 | -100 | 0% |
| sfm_mean | 0.586 | 0.166 | 0.196 | 0% |
| sfm_var | 0.490 | 0.00457 | 0.00402 | 0% |
| flux_p2m | 0.680 | 3.62 | 5.23 | 0% |
| crest_db | 0.420 | 20.2 | 19.3 | 0% |
| mod_peak_db | 0.616 | 5.77 | 7.08 | 0% |
| mod_codec_db | 0.569 | 3.64 | 4.28 | 0% |
| ripple_db | 0.639 | 18.4 | 19.3 | 0% |

| feature set | AUC, model fitted on clean files | AUC, fitted on re-encoded files |
|---|---|---|
| all | 0.638 | 0.688 |
| bandwidth | 0.582 | 0.584 |
| smear | 0.526 | 0.524 |
| codec | 0.687 | 0.634 |

| feature set | leave-sessions-out AUC within phone captures |
|---|---|
| all | 0.606 |
| bandwidth | 0.615 |
| smear | 0.525 |
| codec | 0.549 |

### Per session, clean-fitted 'all' model: mean P(AI)
- 135644a8 'app label' label 1: P(AI) 0.28 over 7
- 16afe47b 'app label' label 0: P(AI) 0.35 over 87
- 16afe47b 'app label' label 1: P(AI) 0.15 over 33
- 2eec88d1 'single song, phone speaker -> ' label 0: P(AI) 0.22 over 1
- 2f5da48d 'app label' label 1: P(AI) 0.99 over 12
- 36cf2d99 'app label' label 0: P(AI) 0.42 over 21
- 4fd86149 'fully AI song, phone speaker -' label 1: P(AI) 0.67 over 2
- 6c9940d0 'human DAW-produced song, phone' label 0: P(AI) 0.02 over 2
- 7fd3b795 'app label' label 1: P(AI) 0.63 over 24
- 81f9d6e3 'app label' label 0: P(AI) 0.38 over 111
- 8b2b4432 'human computer-assisted song, ' label 0: P(AI) 0.09 over 2
- 933e7d2d 'Suno playlist, Mac speakers ->' label 1: P(AI) 0.54 over 118
- bfeddee8 'app label' label 1: P(AI) 0.28 over 17
- e205e9cc 'app label' label 0: P(AI) 0.29 over 120
- e31b0ae9 'AppleMusic:80s+synth' label 0: P(AI) 0.21 over 28
- e3bdb51b 'single song, phone speaker -> ' label 0: P(AI) 0.09 over 2
- e70d7729 'training_data/ai_generated/A M' label 1: P(AI) 0.59 over 2
- e70d7729 'training_data/ai_generated/Bro' label 1: P(AI) 0.52 over 2
- e70d7729 'training_data/ai_generated/Cit' label 1: P(AI) 0.96 over 1
- e70d7729 'training_data/ai_generated/Na ' label 1: P(AI) 0.97 over 1
- e70d7729 'training_data/dj_mixes/[Hallow' label 0: P(AI) 0.52 over 2
- e70d7729 'training_data/human_talent/Lem' label 0: P(AI) 0.51 over 1
- e70d7729 'training_data/human_talent/Ric' label 0: P(AI) 0.59 over 1
- e70d7729 'training_data/human_talent/Wee' label 0: P(AI) 0.57 over 2
- f59eb7d0 'app label' label 1: P(AI) 0.19 over 4
- f82469e7 'Apple music generated playlist' label 0: P(AI) 0.29 over 120
- fd9aaa00 'apple music ai generated songs' label 1: P(AI) 0.30 over 35
