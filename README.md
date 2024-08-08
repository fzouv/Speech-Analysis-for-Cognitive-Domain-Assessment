# Speech-Analysis-for-Cognitive-Domain-Assessment
SLP Dissertation Project 2024, code for replicability


1. OpenSmileFeatureExtraction.py extracts speech features from audio files
2. ArffAppender.py is called from OpenSmileFeatureExtraction to add all features into a csv file
3. paths.ini specifies the paths used in scripts 1 and 2
4. vad_script is a voice activity detector used to segment the audio files to utterance-level
5. preprocess_audio.py was used for some necessary downsampling etc. to use some tools like the vad and the feature extractor
6. basic_classifier.py is my basic logistic regression model used to predict MCI or HC based on speech features (Exp. 1). Also includes a baseline model for comparison.
7. predict_total_scores is the Random Forest regressor predicting the total moca scores of participants based on speech features (Exp.2). It also includes a baseline model for comparison.
8. scores_by_task are the Random Forest regressors that predict the moca scores for each task based on speech features, with fine-tuning based on top 20 features (Exp.3)
9. cross_validation is the script to perform 5-fold cross-validation for Exp. 3
10. correlations computes the Pearson's correlation coefficients between speech features and scores in moca tasks



