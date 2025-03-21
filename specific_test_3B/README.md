# Specific Test III. Image Super-resolution

## Task 3.B

### Data Description

- The dataset comprises 300 strong lensing image pairs at multiple resolutions: high-resolution (HR) and low-resolution (LR)

### Task

- To train a deep learning-based super-resolution algorithm to enhance low-resolution strong lensing images using a limited dataset of real HR/LR pairs collected from HSC and HST telescopes. Implement in PyTorch or Keras.

- For this task, I used techniques like data augmentation, fine-tuning using model of task 3.A, hyper parameters tuning, perceptual loss, Generative adversarial networks.

- I used these models:
  * Super-Resolution Generative Adversarial Network (SRGAN) (Performed Better✅) - file - `gsoc-task-3b.ipynb`
  * FSRCNN fine-tuned on task 3.A (Didn't performed well due to model limitations)
 
### Results

- For SRGAN (Performed Better✅)
<img width="521" alt="Screenshot 2025-03-21 at 11 44 51 PM" src="https://github.com/user-attachments/assets/c4a2085b-c798-46d0-91c2-634d3a62474c" />
<img width="878" alt="Screenshot 2025-03-21 at 11 45 33 PM" src="https://github.com/user-attachments/assets/0f8ef8f6-472d-4ce1-8630-51c9bcb49370" />



- For Fine-Tuned FSRCNN from Task 3.A

<img width="634" alt="Screenshot 2025-03-21 at 11 41 22 PM" src="https://github.com/user-attachments/assets/96a43067-be0f-4670-bc1b-29fa20a7bea7" />
<img width="669" alt="Screenshot 2025-03-21 at 11 44 04 PM" src="https://github.com/user-attachments/assets/8a9cf7c9-c4a2-4f97-8a39-f773758ea4bd" />
<img width="697" alt="Screenshot 2025-03-21 at 11 46 09 PM" src="https://github.com/user-attachments/assets/e2afe7c4-e0c0-4c73-ad6d-b79d05202e61" />




