# Paraphrase identification

## Table of Contents

* [About](#about)
* [Contents](#contents)
* [Installation](#installation)
* [Results](#results)
* [Contact](#contact)

<br>

## About
This projet aims to build Neural Network model to predict if two questions are paraphrases or not using Deep Learning.

## Contents
This project contains:
- A folder for notebooks (Project contains our model results)
- Reporting folder containing the report from this project 
- A folder "scripts" with data_utils module containing ETL and Embedding pipelines + two scripts for the two tested models  (SiameseLSTM and BERTFineTuner)
- Env files (explained after)

## Installation
To use this project, you must make the follow commands:
```
git clone https://github.com/luciegaba/paraphrase-identification.
cd paraphrase-identification
```
If you run the code for BERT Fine-tuning part in Colab, you must do instead:
```
pip install -r requirements.txt
```
If you use conda virtual env:
```
conda env create -f environment.yml
conda activate paraphrase-identification
```
### Results
In this project, we mainly focused on developing a model from scratch to challenge ourselves. We built a Siamese LSTM model for this purpose. Nonetheless, you will see that our performance were not so good due to lack of quality fo data and a potential badly calibrated model.
But we also make a "challenging" model based on Transformers called "ParaBERT": The BERT fine-tuned model can be found [here](https://huggingface.co/luciegaba/ParaBERT). 
See more details about our project in our [report](https://github.com/luciegaba/paraphrase-identification/tree/main/reporting)

### Contact
* [Lucie Gabagnou👸](https://github.com/luciegaba) - Lucie.Gabagnou@etu.univ-paris1.fr
* [Armand L'Huillier👨‍🎓](https://github.com/armandlhuill) - Armand.lHuillier@etu.univ-paris1.fr
* [Yanis Rehoune👨‍🎓](https://github.com/Yanisreh) - Yanis.Rehoune@etu.univ-paris1.fr
* [Ghiles Idris👨‍🎓](https://github.com/ghiles10) - Ghiles.Idris@etu.univ-paris1.fr

