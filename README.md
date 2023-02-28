# 🍔 Food Vision 📷

An end-to-end CNN Image Classification Model which identifies the food in your image.

I exercised using a pre-trained Keras image classification model after that retrained it using the Food101 Dataset.

### Objective:

Make a better model than the DeepFood Paper's model which also trained on the same dataset.

Dataset : Food101

 Model: EfficientNetB0
        EfficientNetB7
        
 Training time : 47 min.
 
 Accuracy: 80.40 %

### How Food-vision build ?

- Downloading Food101 dataset from Tensorflow Dataset Module.
- Knowing dataset: Visualise-Visualise-Visualise
- Setup Mixed Precision
  In order to train model faster setup global dtype policy to mixed_float16(Implementing Mixed   Precision Training)
- Build feature extraction Model
- Fit feature extraction Model
- Load and evaluate checkpoint weights
- Save model and use later on
- Preparing above model for fine-tuning
- Model Callbacks(Minimising resources unnecessary use)
  * Tensorboard Callback: TensorBoard provides the visualization and tooling needed for      
    
                          machine learning experimentation.
  * EarlyStoppingCallback:Used to stop training when a validation loss has stopped reducing.
  * ReduceLROnPlateau : Reduce learning rate when a model is not finding better prediction      
                        than previous epochs.
- Building and Training of a Fine Tuning Model:
  In this, we use pretrained models weights from above model and tweaked it get better         
  results. Architecture : EffficientNetB0
- Evaluating results using latest result.

