
# ML4SCI 2025 GSoC DeepLense Task

This GitHub repository contains three Folders, each of which focuses on a different deep learning task.
All the implementation is in PyTorch.

For all the weights, please <a href="https://www.kaggle.com/models/xanderex/gsoc-deeplense-test-weights/">CLICK HERE</a>

## Common Test: Multi-class Classification

The <a href="https://github.com/xanderex-sid/GSoC_DeepLense/blob/main/common_test/gsoc-commontask-deeplense.ipynb">notebook</a> in `common_test` demonstrates a simple image classification task using a convolutional neural network (Transfer Learning). The dataset used in this notebook is the one provided for common test, which is a collection of strong lensing image. The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance.

#### Results on various Models:

| Model                                       | Epochs | Batch Size | Learning Rate | ROC_AUC   |
| :------------------------------------------ | :----- | :--------- | :------------ | :-------- |
| DenseNet169                                | 15      | 128         | 0.0001        | 0.987      | 
| DenseNet161                                 | 17     | 128         | 0.0001        | 0.986      |     
| Ensemble                                    | -      | -          | -             | 0.988      | 


## Specific Test 3: Image Super-resolution

### Task 3.A:

The <a href="https://github.com/xanderex-sid/GSoC_DeepLense/blob/main/specific_test_3A/gsoc-task-3a.ipynb">notebook</a> in `specific_test_3A` trains a deep learning-based super resolution algorithm, specifically, Fast Super-Resolution Convolutional Neural Network (FSRCNN) to upscale low-resolution strong lensing images using the provided high-resolution samples as ground truths. Laplacian Pyramid Super-Resolution Network (LapSRN) was also used for training, but didn't performed well.

#### Evaluation Results of Models on Validation Dataset:

| Model                                       | SSIM  | PSNR       | MSE     | L1 Loss  |
| :------------------------------------------ | :---- | :----------| :----   | :------- |
| FSRCNN (Performed Better✅)                 | 0.9563 | 38.8625 dB | 2e-9    | 0.000035 | 
| LapSRN                                      | 0.7568 | 33.2988 dB  | 0.000001 | 0.001 |  



### Task 3.B:

The <a href="https://github.com/xanderex-sid/GSoC_DeepLense/blob/main/specific_test_3B/gsoc-task-3b-SRGAN.ipynb">notebook</a> `gsoc-task-3b-SRGAN.ipynb` in `specific_test_3B` trains a Super-Resolution Generative Adversarial Network (SRGAN) algorithm to enhance low-resolution strong lensing images using a limited dataset of real HR/LR pairs collected from HSC and HST telescopes. It uses transfer learning, data augmentation and min-max normalization.

I also used FSRCNN (from task 3.A) for fine-tuning in the <a href="https://github.com/xanderex-sid/GSoC_DeepLense/blob/main/specific_test_3B/gsoc-task-3b-fine-tuned-fsrcnn.ipynb">notebook</a> `gsoc-task-3b-fine-tuned-fsrcnn.ipynb` and performed experiments like using perceptual loss and L1 loss together or using L1 loss only instead of MSE loss.

#### Evaluation Results of Models on Validation Dataset:

| Model                                       | SSIM  | PSNR       | MSE     | L1 Loss  |
| :------------------------------------------ | :---- | :----------| :----   | :------- |
| SRGAN                                        | 0.6521 | 28.545 dB | 0.001557    | 0.034736 | 
| FSRCNN (Fine-tuned On Test 3.A)                  | 0.7179 | 27.7687 dB  | 0.000000026 | 0.000083 | 

## Usage

#### 1) Setup 

Clone the Repository
```bash
$ git clone https://github.com/xanderex-sid/GSoC_DeepLense.git
```
Move to the directory to access the notebooks
```bash
cd GSoC_DeepLense
```
- For **Common Test**:
    - `gsoc-commontask-deeplense.ipynb` notebook is used.
- For **Specific Test 3.A**:
    - `gsoc-task-3a.ipynb` notebook is used.
- For **Specific Test 3.B**:
    - `gsoc-task-3b-SRGAN.ipynb` notebook is used for SRGAN model.
    - `gsoc-task-3b-fine-tuned-fsrcnn.ipynb` notebook is used for fine-tuning specific task 3.A.

#### 2) Dataset Directory

Give the path of dataset. Eg:
```python
root_dir = '[Your Dataset Path]'
dir_no_sub = root_dir+'/no_sub/' # Path to folder having data with no substructure 
dir_sub = root_dir+'/sub/' # Path to folder having data containing substructure
```
#### 3) Hyperparameters Setting
Use `CFG` class to change the hyperparameters
- For **Common Test**:
```python
class CFG:
    lr = 0.0001
    batch_size = 128
    num_classes = 3
    target_col="target"
    epochs = 15
    seed = 42
    num_workers = 2
    transform = False
    weight_decay = 1e-2
    num_workers=2
    train=True
    debug=False
    metric_type="roc_auc"
    scheduler_type = "StepLR"
    optimizer_type = "Adam"
    loss_type = "CrossEntropyLoss"
    max_grad_norm = 1000
    lr_max = 4e-4
    epochs_warmup = 1.0
    model_name = "densenet169"
 ```

- For **Specific Test III**:
```python
class CFG:
    def __init__(self):
        self.lr_folder = "/kaggle/input/gsoc-dataset-task3-a/Dataset/LR"
        self.hr_folder = "/kaggle/input/gsoc-dataset-task3-a/Dataset/HR"
        self.batch_size = 16
        self.val_batch_size = 16
        self.num_workers = 4
        self.train_size = 0.7
        
        self.model_d = 56
        self.model_s = 12
        self.model_m = 4
        
        self.lr_init = 0.001
        self.epochs = 75
        self.weights_fn = "best_fsrcnn_model.pth"
 ```
 #### 4) Augmentation
 For augmentation use `get_transform` function
 ```python 
 def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(*CFG.size),
            A.Rotate(limit=20),
            A.InvertImg(), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(*CFG.size),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
 ```
 
 #### 5) Training and Evaluation
 For training and evaluation use `Train` class
 ```python
 def main():
    
    if CFG.train: 
        # train
        train = Train(CFG) # Configuration class
        train.train_loop(train_df, val_df) # Dataframe for training and evaluation
        
 if __name__=='__main__':
    main()
```

## Dependencies

To run the notebooks, you will need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- PyTorch

You can install these dependencies using pip or conda.
