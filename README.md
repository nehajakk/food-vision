# 🍔 Food Vision 📷

an end-to-end CNN Image Classification Model which identifies the food in your image.
I exercised using a pre-trained Keras image classification model after that retrained it using the Food101 Dataset.
Objective:
Make a better model than the DeepFood Paper's model which also trained on the same dataset.
 Dataset : Food101
 Model: EfficientNetB0
 EfficientNetB7
 Accuracy:

How Food-vision build ?
Downloading Food101 dataset from Tensorflow Dataset Module.
Knowing dataset: Visualise-Visualise-Visualise
In order to train model faster setup global dtype policy to mixed_float16(Implementing Mixed Precision Training)
Model Callbacks(Minimising resources unnecessary use)
Tensorboard Callback: TensorBoard provides the visualization and tooling needed for machine learning experimentation.
EarlyStoppingCallback:Used to stop training when a validation loss has stopped reducing.
ReduceLROnPlateau : Reduce learning rate when a model is not finding better prediction than previous epochs.
Building a Fine Tuning Model:
This part tool the longest. In Deep Learning, you have to know which nob does what. Once yoy get experienced you'll what nobs you should turn to get the results you want. Architecture : EffficientNetB0

