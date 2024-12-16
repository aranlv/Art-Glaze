# Image Glazing for AI-Proof Artwork Protection  
*Final Project - COMP7116001: Computer Vision*  

**By:**  
- Aretha Natalova Wahyudi - 2602114605  
- Axel Nino Nakata - 2602050671  
- Jessica Lynn Wibowo - 2602053490  

*School of Computer Science, Universitas Bina Nusantara, 2024*

---

## Overview  

**Image Glazing for AI-Proof Artwork Protection** is a research-driven solution to protect digital artworks from unauthorized use in AI training models. The system applies imperceptible perturbations to artworks, reducing the ability of AI systems to replicate the original artistic style without compromising human-perceived image quality.  

This project integrates advanced image processing, local feature descriptors, and machine learning techniques to optimize and evaluate the glazing process, providing a comprehensive framework for ethical AI development and intellectual property protection.  

---

## Objectives  

- Develop a glazing technique to protect artworks from unauthorized AI model training.  
- Ensure minimal visual impact on the original image while effectively disrupting AI style replication.  
- Implement and evaluate feature extraction, optimization, and classification methods to enhance the glazing process.  

---

## Methodology  

### Image Preprocessing  

- **Bilateral Filter:** Smoothens images while preserving edge details, maintaining critical visual features.  
- **Unsharp Masking:** Enhances edge contrast for better feature clarity.  
- **Normalization:** Adjusts image mean and standard deviation to match the ImageNet dataset, optimizing model compatibility.  

### Feature Extraction  

- **Global Features:** Extracted using VGG-19 to capture high-level image representations.  
- **Local Descriptors:**  
  - **ORB (Oriented FAST and Rotated BRIEF):** Efficiently extracts binary descriptors to identify key image features.  
  - **SIFT (Scale-Invariant Feature Transform):** Detects scale- and rotation-invariant features for robust feature matching.  

### Optimization  

- **Perturbation Calculation:**  
  Gradient-based optimization (Adam Optimizer) is used to generate imperceptible perturbations that reduce AI feature extraction accuracy.  

- **Loss Functions:**  
  - **Feature Loss:** Minimizes differences in high-level feature representations.  
  - **Perceptual Loss (LPIPS):** Preserves perceptual image quality.  
  - **Descriptor-Based Loss:** Reduces ORB and SIFT keypoint matching to disrupt AI feature recognition.  

- **Total Loss:**  
  \[
  \text{Total Loss} = \text{Feature Loss} + (\text{LPIPS Loss - } p) + \text{ORB Loss} + \text{SIFT Loss}
  \]  
  where \( p \) is a perceptual similarity threshold, ensuring minimal visual differences.  

### Classification  

A Support Vector Machine (SVM) is used to classify glazed and unglazed images based on extracted features:  
- **Color Histogram**  
- **Histogram of Oriented Gradients (HOG)**  
- **SIFT + ORB Descriptors**  

---

## Evaluation  

### Qualitative Analysis  

The technique was evaluated using Stable Diffusion XL (SD-XL) to assess style replication accuracy:  
- **Unglazed Images:** AI accurately replicates the artistic style.  
- **Glazed Images:** Style replication significantly reduced, with the image remaining visually intact for human perception.  

### Quantitative Analysis  

1. **Keypoint Matching:**  
   - ORB and SIFT descriptors showed ~60% matching reduction, demonstrating successful disruption of local feature recognition.  

2. **Classification:**  
   - **Color Histogram** and **SIFT + ORB** features achieved the best balance between precision, recall, and F1-score.  
   - Overall accuracy remained at ~50% due to dataset size limitations.  

---

## Results  

- The glazing technique effectively disrupts AI style replication while maintaining aesthetic integrity.  
- Keypoint matching reduction confirms the disruption of local feature detection.  
- Classification models identify glazed images with reasonable precision, highlighting the importance of feature selection.  

---

## Limitations  

- Limited dataset size impacts classification performance.  
- Subjective evaluation of glazing effectiveness requires professional artist input.  
- Exploration of advanced deep learning classifiers and hyperparameter tuning could further optimize results.  

---

## References  

1. Molla, M. (2024). *AI in Creative Arts: Advancements and Innovations in Artificial Intelligence*. DOI: [10.48175/ijarsct-19163](https://doi.org/10.48175/ijarsct-19163).  
2. Shan, S. et al. (2023). *Glaze: Protecting artists from style mimicry by Text-to-Image models*. 32nd USENIX Security Symposium.  
3. Zhong, H. et al. (2023). *Copyright protection and accountability of generative AI*. DOI: [10.1145/3543873.3587321](https://doi.org/10.1145/3543873.3587321).  
