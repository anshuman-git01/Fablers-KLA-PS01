All submissions are made through the i4C hackathon portal. You must submit a PPT/PDF using the provided Idea Submission Template AND a GitHub repository link. Here is exactly what each must contain:  
**Component 1 — PPT/PDF Submission (Using Idea Submission Template)**  
Use the provided Hackathon Idea Submission Template. Fill in each slide as follows:

| Template Slide | What to Fill In |
| :---- | :---- |
| Slide 1: Team Details | Team name, member names, roles, college name, contact details. |
| Slide 2: Problem Statement Addressed | Select ‘AI-Based Restoration of Degraded Images.’ Describe in your own words why this problem matters in semiconductor manufacturing. |
| Slide 3: Idea Description | Your key concept and approach – what type of AI model did you choose? Why? How does it address all 3 degradation types (speckle, Gaussian, super-resolution)? |
| Slide 4: Proposed Solution | Detailed solution – model architecture, training strategy, loss function design, data augmentation approach. Include a system/pipeline diagram. |
| Slide 5: Innovation & Uniqueness | What makes your approach different? Did you design a novel loss function? A unique data augmentation strategy? A faster inference pipeline? |
| Slide 6: Results | SSIM, pSNR, LPIPS scores on your test split. Before/after image comparisons (degraded input → your restored output → ground truth). Confusion-free visual evidence that your model works. |
| Slide 7: Technology & Feasibility | Tech stack used (PyTorch/TensorFlow/other), hardware used for training (GPU type, cloud platform), training time, model size, inference time per image. |
| Slide 8: GitHub & Video Link | GitHub repository link (mandatory). Video link showing your model running (optional but recommended). |
| Slide 9: References | Research papers, datasets, tools referenced. |

FILE FORMAT: Save as PDF before uploading. File naming convention: TeamName\_KLA\_PS01 (e.g., VisionForge\_KLA\_PS01.pdf). Maximum 8-9 slides. Remove the instruction slide.  
**Component 2 — GitHub Repository (Mandatory)**  
Your GitHub repository must be public and must contain:

| \# | Repository Content | Details |
| :---- | :---- | :---- |
| 1 | README.md | Complete setup instructions. A reviewer must be able to clone your repo and run inference from the README without contacting you. |
| 2 | Evaluation Script (standalone .py) | A Python script (NOT a Jupyter notebook) that accepts: (a) path to test images directory, (b) path to output directory. It loads your trained model, runs inference on all input images, and writes restored outputs to the specified directory. Must run without manual edits. |
| 3 | Training Script | Python script or Jupyter notebook that reproduces your training process from scratch. |
| 4 | Trained Model Weights | Your final trained model file (in a format your evaluation script can load — .pt, .onnx, .h5, etc.). Must be downloadable (use Git LFS or link to Google Drive/HuggingFace if file is large). |
| 5 | Restored Test Outputs | A folder containing your model’s output on the test set — the actual restored images your model produced. |
| 6 | requirements.txt | Complete pip freeze output from your training environment. Required for reproducibility. |

**CRITICAL:** The evaluation script is the most important file in your repository. It will be used AS-IS by KLA’s benchmarking team to measure your model’s quality scores and inference time on the H100 GPU. If your script does not run without manual edits, your submission cannot be benchmarked and unscored submissions cannot win. Test your script on a fresh machine before submitting.

