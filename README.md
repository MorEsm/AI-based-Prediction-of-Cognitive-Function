# AI-based Prediction of Cognitive Function
In this study, we used a deep learning approach to evaluate the predictive power of the functional connectome during various states (resting state, movie-watching, and n-back) on episodic memory and working memory performance.


M. Esmaeili, E.B. Bjørkeli, R. Pedersen, F. Falahati, J. Johansson, K. Nordin, N. Karalija, L. Bäckman, L. Nyberg, A. Salami, Brain-Cognitive Gaps in relation to Dopamine and Health-related Factors: Insights from AI-Driven Functional Connectome Predictions, eLife Sciences Publications, Ltd, 2025. https://doi.org/10.7554/eLife.104053.1


# DenseNet + Attention Regression on Functional Connectivity Matrices

Predicts a continuous target (age, episodic-memory score, etc.) from square
functional-connectivity (FC) matrices using a DenseNet-style CNN with
Enhanced Residual Blocks (ERB) and High-Frequency Attention Blocks (HFAB).

## Usage

```bash
pip install -r requirements.txt

python train_densenet_fc_age.py \
    --data-dir /path/to/fc_matrices \
    --labels-csv /path/to/labels.csv \
    --subject-id-col SubjectID \
    --target-col AGE \
    --matrix-size 273 \
    --mat-key IMG_temp \
    --output-dir ./runs/age_v1
```


