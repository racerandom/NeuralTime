# BERT-based Temporal Relation Classifier

## Overview
This project includes several BERT-Based Temporal relation classifiers for BCCWJ-Timebank[Asahara 2013].

- Pair-wised model  
- Multi-task learning model  
- Source event-centric model  

I've placed corpus in the data folder.


## Pair-wise model 
Pair-wise models perform 'DCT', 'T2E', 'E2E', 'MAT' as independent classifiers.

[document-level 5-fold cross-validation]  

Train:
> python multiTaskClassifier.py \  
> --task 'DCT' \  # please input 'DCT', 'T2E', 'E2E' or 'MAT'  
> --pre $BERT\_DIR \ # pre-trained BERT dir  
>  --model\_dir $MODEL\_DIR \ # new dir to save model  
> --batch 16 \  
> --epoch 5 \ # fine-tuning epochs e.g. 3~7  
> --do\_train # training model
 
~ print(test) ~

Test:
> python multiTaskClassifier.py \  
> --task 'DCT' \  # please input 'DCT', 'T2E', 'E2E' or 'MAT'  
> --model\_dir $MODEL\_DIR \ # model dir to load

## Multi-Task model 

Pair-wise models jointly training 'DCT', 'T2E', 'E2E', 'MAT' batchs in one classifier.

[document-level 5-fold cross-validation]

Train:
> python multiTaskClassifier.py \  
> --task 'ALL' \  # The model loads 'DCT', 'T2E', 'E2E', 'MAT' batchs in one sequence  
> --pre $BERT\_DIR \ # pre-trained BERT dir  
> --model\_dir $MODEL\_DIR \ # new dir to save model  
> --batch 16 \  
> --epoch 5 \ # fine-tuning epochs e.g. 3~7  
> --do\_train # training model

Test:
> python multiTaskClassifier.py \  
> --task 'ALL' \  
> --model\_dir $MODEL\_DIR \ # model dir to load

## Source Event-centric model

This is also a joint model to train 'DCT', 'T2E', 'E2E', 'MAT'.I've not create a API for this model, but you can replace 'PRETRAIN\_BERT\_DIR' with your pre-trained BERT model in 'eventCentreClassifier.py', then run it. Currently it use one data split, not 5-fold CV.

Run:
> Python eventCentreClassifier.py

## Results
TBD

## Required Packages
pytorch=1.0.0  
pytorch-pretrained-bert=0.6.2  
mojimoji  
tqdm  
pyknp  
jumanpp 

## Reference
Temporal Relation Classification:  
BCCWJ-Timebank:  
BERT:   
Multi-Task Learning:   

