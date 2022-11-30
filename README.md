# Social Intimacy Analysis

Team members: ABHIJIT MAHAPATRA, ​
ABHINAND NASARI, ​
KUNAL VUDATHU, ​
LAKHAN KAMIREDDY, ​
SUJITH JAYASEELAN, ​
SWARNABHA ROY

## Intimacy Score Predictor code

### Overview
The code exists in NLP Project_initimacy_score.ipynb. This can be run independently in colab. The code should run the best model against semeval dataset. Colab Outputs are already generated for the best model. If you want to play around, please look at training section to update config or hyper parameters.

### Training
Training arguments and hyperparameters can be configured via `Args` and `Config`.
`Args` hold parameters about the model and dataset paths:
```python
class Args:
    model_name # Model Name used for saving output
    pre_trained_model_name_or_path # Pretrained model name on 🤗
    train_path # Path to train dataset .txt file
    val_path # Path to validation dataset .txt file
    test_path # Path to test dataset .txt file
    model_saving_path # Path to save model outputs
```

`Config` holds parameters for training:
```python
class Config:
    max_epochs # Number of epochs for training
    lr # Learning rate 
    batch_size # Batch size
    cuda # Enable CUDA
    max_len # Maximum length of tweets/ posts
    warmup_ratio # Warmup for percentage of total training steps
    weight_decay # Weight Decay
    gradient_accumulation_steps # Accumulate gradients over steps/batches
    adam_epsilon # Epsilon to use in Adam optimizer
```

### Best Model Results
```
model_name = 'xlnet-base-cased'
best_val_loss: 0.13466132375722012
best_test_loss: tensor(0.0795, device='cuda:0')
best_test_pearsonr: 0.6771716153790823
```
## Visualizations code
Intimacy vs topic.ipynb for visualization for gender vs topic based analysis, The colab notebook can be run independently.

## Gender prediction code
Gender_detection.ipnyb contains the code to train-test the Gender prediction, the notebook can be run independently.
