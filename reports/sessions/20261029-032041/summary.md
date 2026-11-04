# App recordings scored with the installed models

Models: 10 s `20261027-005541`, 5 s `20261027-014414`. 22 saved sessions, 9 labelled as a whole, 2 with clip labels; 0 of them were in the models' training data.

Scores follow the app: P = mean of the two models' probabilities, each the sigmoid of its mean window logit over the whole recording. `AI win` is the share of 10 s windows the 10 s model scores above 0.5. `on-device` is what the app computed while recording (blank where an older model did), and `Δlogit` the mean |difference| between the app's 10 s window logits and this re-scoring when the same model made both.

## Sessions

| session | dur | label | clip labels | win | P10 | P5 | P | AI win | verdict | correct | on-device P10/P5 | Δlogit | in training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026-173854-615 | 162 s | nan |  | 31 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-174241-226 | 19 s | nan |  | 2 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-174445-783 | 113 s | nan |  | 21 | 0.00 | 0.01 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-174733-430 | 58 s | nan |  | 10 | 0.00 | 0.01 | 0.01 | 0.00 | human |  |  | - |  |
| 20261026-175308-869 | 17 s | nan |  | 2 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-175346-559 | 50 s | nan |  | 8 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-175437-627 | 503 s | nan |  | 99 | 0.00 | 0.00 | 0.00 | 0.03 | human |  |  | - |  |
| 20261026-180522-743 | 204 s | nan |  | 39 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-231058-378 | 38 s | nan |  | 1 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-231146-468 | 614 s | nan |  | 121 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261026-233640-561 | 3600 s | ai_assisted |  | 718 | 0.00 | 0.00 | 0.00 | 0.00 | human |  |  | - |  |
| 20261027-072022-685 | 202 s | ai |  | 39 | 0.01 | 0.03 | 0.02 | 0.03 | human | **no** | 0.01/0.03 | 0.23 |  |
| 20261027-072607-410 | 105 s | ai |  | 19 | 0.01 | 0.48 | 0.25 | 0.10 | human | **no** | 0.02/0.43 | 0.67 |  |
| 20261027-072803-566 | 713 s | ai |  | 141 | 1.00 | 1.00 | 1.00 | 0.87 | ai | yes | 0.99/1.00 | 1.70 |  |
| 20261027-075552-626 | 513 s | ai_assisted |  | 101 | 0.00 | 0.00 | 0.00 | 0.25 | human |  | 0.00/0.00 | 0.74 |  |
| 20261027-092215-278 | 341 s | ai |  | 0 | - | - | - | - | no windows |  | -/0.56 | - |  |
| 20261027-145812-403 | 1564 s | nan |  | 0 | - | - | - | - | no windows |  | 0.00/0.06 | - |  |
| 20261027-152432-049 | 57 s | human |  | 10 | 0.00 | 0.01 | 0.01 | 0.00 | human | yes | 0.00/0.02 | 0.11 |  |
| 20261027-152534-433 | 3326 s | human |  | 664 | 0.00 | 0.00 | 0.00 | 0.05 | human | yes | 0.00/0.00 | 0.44 |  |
| 20261027-162106-372 | 3600 s | human |  | 718 | 0.00 | 0.00 | 0.00 | 0.06 | human | yes | 0.00/0.00 | 0.45 |  |
| 20261027-174709-008 | 3600 s | nan | {"ai": 33, "human": 87} | 718 | 0.00 | 0.03 | 0.02 | 0.13 | human |  | 0.00/0.04 | 0.52 |  |
| 20261028-080000-418 | 289 s | nan | {"ai": 1, "human": 1} | 56 | 0.01 | 0.04 | 0.03 | 0.23 | human |  | 0.02/0.05 | 1.16 |  |

## Labelled clips (30 s)

Clips take their own label where the app has one, else the recording's. Metrics use the headline P per clip; `window AUC` uses single 10 s windows of the 10 s model.

### all labelled clips: 537 clips (319 human, 137 ai_assisted, 81 ai)

| score | AUC | AI caught @0.5 | human flagged @0.5 |
|---|---|---|---|
| headline | 0.739 | 25/69 | 28/319 |
| 10 s model | 0.769 | 26/69 | 15/319 |
| 5 s model | 0.733 | 26/69 | 44/319 |
| window (10 s) | 0.746 | 146/403 | 154/1908 |

AI-assisted clips (137): median P 0.00, 2 of 137 called AI.

### clips not in training: 537 clips (319 human, 137 ai_assisted, 81 ai)

| score | AUC | AI caught @0.5 | human flagged @0.5 |
|---|---|---|---|
| headline | 0.739 | 25/69 | 28/319 |
| 10 s model | 0.769 | 26/69 | 15/319 |
| 5 s model | 0.733 | 26/69 | 44/319 |
| window (10 s) | 0.746 | 146/403 | 154/1908 |

AI-assisted clips (137): median P 0.00, 2 of 137 called AI.

### Per session and label

| session | label | clips | correct | median P |
|---|---|---|---|---|
| 20261027-072022-685 | ai | 7 | 0 | 0.02 |
| 20261027-072607-410 | ai | 4 | 0 | 0.28 |
| 20261027-072803-566 | ai | 24 | 23 | 1.00 |
| 20261027-092215-278 | ai | 12 | 0 | nan |
| 20261027-152534-433 | human | 111 | 106 | 0.00 |
| 20261027-162106-372 | human | 120 | 113 | 0.00 |
| 20261027-174709-008 | ai | 33 | 2 | 0.01 |
| 20261027-174709-008 | human | 87 | 71 | 0.02 |
| 20261028-080000-418 | ai | 1 | 0 | 0.04 |
| 20261028-080000-418 | human | 1 | 1 | 0.00 |
