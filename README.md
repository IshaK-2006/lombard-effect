# Lombard Effect: Exploratory Acoustic Analysis

## Overview

This repository presents an exploratory analysis of the **Lombard effect**, the phenomenon in which speakers increase their vocal effort in response to elevated background noise. The project focuses on changes in vocal intensity and pitch under controlled masking noise conditions.

The experimental data were collected as part of a **group project during the 10th edition of the Bioacoustics Winter School** held at **Jean Monnet University**. This repository and accompanying term paper document **only the author’s individual analytical contributions and interpretations**, not full credit for data acquisition.

---

## Authorship and Contribution Statement

- **Data collection:** Conducted collaboratively as part of a group experiment during the winter school  
- **Analysis, visualization, and interpretation:** Performed independently by the author  
- **Code:** All Python scripts in this repository were written by the author  
- **Figures and results:** Generated solely from the author’s analysis pipeline  

This repository is intended as a transparent record of individual learning, methodology, and scientific reasoning derived from a collaborative experimental setting.

---

## Research Question

How does vocal output change as a function of increasing background noise level, and which acoustic features show the most robust response under constrained experimental conditions?

The analysis emphasizes **relative changes** in vocal output rather than absolute calibrated SPL values.

---

## Experimental Methodology (Summary)

- Six participants were exposed to broadband white noise at multiple sound pressure levels (Participant identifiers (P01–P06) are anonymized and do not correspond to any personally identifiable information)
- White noise was generated using Audacity and delivered through headphones.
- Participants were instructed to imagine a listener positioned approximately 5 meters away and to repeatedly vocalize the utterance *“ohho”* for 20 seconds per condition.
- Noise levels were presented in randomized order to reduce order effects.
- Vocalizations were recorded using a fixed microphone setup.
- Acoustic features were extracted from recordings and analyzed using Python.

---

## Data and Ethics Statement

Raw audio recordings, including human speech, are **not publicly shared** due to ethical considerations related to human subjects and biometric data.  

Processed numerical data (e.g., SPL and pitch measurements) sufficient to reproduce the analysis are provided in CSV format. The absence of raw audio does not affect the interpretability of the presented results.

---

## Analysis Overview

The analysis includes:

- Extraction of broadband SPL values
- Baseline normalization relative to low-noise conditions
- Visualization of vocal intensity changes using box plots
- Exploratory pitch analysis

Key scripts include:

- `analysis/decibel_level.py` – SPL extraction and preprocessing  
- `analysis/relative_dBlevel.py` – Baseline normalization and visualization  
- `analysis/pitch.py` – Exploratory pitch analysis  

---

## Results Summary

- Vocal intensity shows a clear increase with increasing background noise when analyzed relative to individual baselines.
- The Lombard effect is consistently observed across participants despite inter-subject variability.
- Pitch-related changes do not exhibit a clear or systematic trend under the present experimental conditions.

Figures and plots are included in the `results/` directory.

---

## Limitations

This study is exploratory in nature. Key limitations include:

- Small sample size (n = 6)
- Absence of absolute SPL calibration
- Use of imagined communication distance
- Broadband SPL analysis without frequency resolution
- Single trial per condition
- Fixed recording geometry

These limitations are explicitly acknowledged to avoid overinterpretation.

---

## Future Work

Potential extensions include increasing participant numbers, stratifying subjects by age and gender, varying the spectral properties of masking noise, performing frequency-resolved analyses, and incorporating calibrated sound delivery with repeated trials.

---

## Tools and Dependencies

- Python
- pandas
- matplotlib
- PyTorch (for the ML regression model)
- Audacity (used during data acquisition only)

---

## Project Status

Completed as an exploratory and educational project demonstrating experimental acoustics, ethical data handling, and reproducible analysis.

- Added a PDF file in the `results` folder that explains the regression graphs, including model predictions, feature effects, and visualization of the Lombard effect.
- Provides detailed insights into the Machine Learning extension and how the model predicts baseline-normalized vocal intensity (ΔSPL).


## Machine Learning Extension

- Added a **Machine Learning module**: a multivariate regression model implemented in **PyTorch** to predict **baseline-normalized vocal intensity (ΔSPL)** based on **environmental noise level** and **speaker identity**.
- Demonstrates **feature engineering**, **normalization-aware modeling**, and **supervised learning** applied to experimental acoustic data.
-  Added a PDF file in the `results` folder that explains the regression graphs, including model predictions, feature effects, and visualization of the Lombard effect.
- Provides detailed insights into the Machine Learning extension and how the model predicts baseline-normalized vocal intensity (ΔSPL).
- Provides a foundation for exploring the Lombard effect with predictive modeling and potential applications in **speech enhancement** and **voice-controlled systems**.

