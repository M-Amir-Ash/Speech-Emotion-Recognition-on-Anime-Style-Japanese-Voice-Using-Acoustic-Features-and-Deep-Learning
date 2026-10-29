# Audio Quality Report

Twenty-one local-review samples were selected from the verified 1,969-record cohort: three per assigned class and one per split per class. Source-group diversity and three technical-screening strata were used.

Metrics are descriptive signal measurements. Silence uses a -45 dBFS frame threshold; clipping counts samples at or above 0.999 full scale; spectral flatness is a background/noise proxy. The technical screening score averages audibility, non-silence, unclipped signal, tonal structure, historical language probability, and the available speaker-overlap proxy. It cannot verify emotion, intelligibility, copyright status, or human-perceived quality. Labels are inherited pseudo-labels and require lecturer review.

| ID | Assigned label | Split | Duration (s) | RMS dBFS | Clipping | Silence | Stratum |
|---|---|---|---:|---:|---:|---:|---|
| AQ01 | Angry | train | 2.29 | -21.8 | 0.00000 | 0.000 | higher_screened |
| AQ02 | Angry | validation | 1.86 | -21.7 | 0.00000 | 0.000 | middle_screened |
| AQ03 | Angry | test | 1.11 | -21.9 | 0.00000 | 0.000 | lower_screened |
| AQ04 | Disgust | train | 1.62 | -25.0 | 0.00000 | 0.000 | higher_screened |
| AQ05 | Disgust | validation | 1.33 | -28.5 | 0.00000 | 0.000 | middle_screened |
| AQ06 | Disgust | test | 0.98 | -30.2 | 0.00000 | 0.000 | lower_screened |
| AQ07 | Fear | train | 3.02 | -26.8 | 0.00000 | 0.000 | higher_screened |
| AQ08 | Fear | validation | 1.04 | -28.4 | 0.00000 | 0.000 | middle_screened |
| AQ09 | Fear | test | 1.94 | -26.6 | 0.00000 | 0.000 | lower_screened |
| AQ10 | Happy | train | 1.47 | -24.8 | 0.00000 | 0.000 | higher_screened |
| AQ11 | Happy | validation | 4.68 | -28.2 | 0.00000 | 0.035 | middle_screened |
| AQ12 | Happy | test | 3.02 | -24.5 | 0.00000 | 0.000 | lower_screened |
| AQ13 | Neutral | train | 1.20 | -26.2 | 0.00000 | 0.000 | higher_screened |
| AQ14 | Neutral | validation | 0.97 | -24.8 | 0.00000 | 0.000 | middle_screened |
| AQ15 | Neutral | test | 1.06 | -26.0 | 0.00000 | 0.000 | lower_screened |
| AQ16 | Pleasant_Surprise | train | 3.21 | -22.9 | 0.00000 | 0.010 | higher_screened |
| AQ17 | Pleasant_Surprise | validation | 1.14 | -30.0 | 0.00000 | 0.000 | middle_screened |
| AQ18 | Pleasant_Surprise | test | 0.91 | -25.2 | 0.00000 | 0.000 | lower_screened |
| AQ19 | Sad | train | 2.13 | -30.0 | 0.00000 | 0.000 | higher_screened |
| AQ20 | Sad | validation | 1.77 | -34.0 | 0.00000 | 0.038 | middle_screened |
| AQ21 | Sad | test | 1.01 | -33.2 | 0.00000 | 0.000 | lower_screened |

The audio itself is withheld. Authorized reviewers use the matching `review_id` in the access-controlled local package. Figures contain only signal-derived waveform and Mel-spectrogram representations.
