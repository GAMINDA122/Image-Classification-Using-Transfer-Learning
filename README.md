# Image Classification Using Transfer Learning

This project focuses on **binary image classification (Cat vs. Dog)** and demonstrates the performance difference between a **custom Convolutional Neural Network (CNN)** and a **transfer learning approach using Xception**. It highlights how pretrained models significantly improve accuracy, convergence speed, and generalization in real-world computer vision tasks.

---

## Project Overview

The project implements two approaches for image classification:

1. **Custom CNN Model**
   - Built from scratch using convolutional, pooling, and dense layers
   - Achieved **79.6% accuracy**
   - Demonstrates fundamental CNN architecture design and training

2. **Transfer Learning with Xception**
   - Uses a pretrained Xception model (ImageNet weights)
   - Achieved **95.6% accuracy in a single epoch**
   - Demonstrates the efficiency and effectiveness of transfer learning

The comparison clearly shows how pretrained models outperform custom architectures in both accuracy and training efficiency.

---

## Key Concepts Demonstrated

- Convolutional Neural Networks (CNNs)
- Transfer Learning with Xception
- Binary Classification with Sigmoid Activation
- Feature Scaling and Normalization
- Data Augmentation for model generalization
- Model Evaluation (Accuracy, Precision, Recall)
- Performance comparison between custom and pretrained models

---

## Technologies Used

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- OpenCV  
- Pickle  

---

## Skills Applied

- Machine Learning  
- Deep Learning  
- Computer Vision  
- CNN Architecture Design  
- Transfer Learning  
- Data Augmentation Techniques  
- Model Evaluation and Visualization  
- Problem Solving  

---

## Repository Structure
```text
Image-Classification-Using-Transfer-Learning/
├── CNN.ipynb
└── README.md
```

---

## Workflow Summary

1. Load and preprocess image datasets
2. Apply feature scaling and normalization
3. Perform data augmentation to improve generalization
4. Train a custom CNN model and evaluate performance
5. Train a transfer learning model using Xception
6. Compare accuracy, training time, and convergence behavior
7. Visualize results using accuracy and loss plots

---

## Results

- **Custom CNN Accuracy:** 79.6%  
- **Xception Transfer Learning Accuracy:** 95.6% (1 epoch)  

The results demonstrate that transfer learning dramatically improves performance while reducing training time and computational cost.

---

## Purpose

This project showcases practical application of **transfer learning in image classification**, making it suitable for real-world AI systems where accuracy, efficiency, and scalability are critical.

---

## License

This repository is open-source and intended for educational, research, and portfolio use.






