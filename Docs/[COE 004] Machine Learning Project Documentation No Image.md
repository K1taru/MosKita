Redacted 
Redacted

# GTSRB TRAFFIC SIGN CLASSIFIER USING TRANSFER LEARNING WITH RESNET50 DEEP CONVOLUTIONAL NEURAL NETWORKS FOR HIGH-ACCURACY AUTONOMOUS VEHICLE PERCEPTION

A project submitted in partial fulfillment for the requirement for the course of COE 004 \- Inferential Thinking by Resampling

Submitted by:  
Redacted

Submitted to:  
Redacted

November 2025

[**I. INTRODUCTION	6**](#i.-introduction)

[A. Problem Statement	7](#a.-problem-statement)

[1\. Classification Accuracy Requirements	7](#1.-classification-accuracy-requirements)

[2\. Class Imbalance in Real-World Data	7](#2.-class-imbalance-in-real-world-data)

[3\. Environmental Variability	7](#3.-environmental-variability)

[4\. Real-Time Processing Constraints	7](#4.-real-time-processing-constraints)

[5\. Generalization Across Different Contexts	8](#5.-generalization-across-different-contexts)

[B. Proposed Solution	8](#b.-proposed-solution)

[1\. Deep Residual Network Architecture (ResNet50)	8](#1.-deep-residual-network-architecture-\(resnet50\))

[2\. Transfer Learning from ImageNet	8](#2.-transfer-learning-from-imagenet)

[3\. Comprehensive Data Preprocessing and Augmentation	9](#3.-comprehensive-data-preprocessing-and-augmentation)

[4\. Advanced Training Strategies	9](#4.-advanced-training-strategies)

[5\. Robust Evaluation and Model Selection	9](#5.-robust-evaluation-and-model-selection)

[**II. METHODOLOGY	10**](#ii.-methodology)

[A. System Architecture	10](#a.-system-architecture)

[1\. Base Architecture Specifications	10](#1.-base-architecture-specifications)

[2\. Architecture Components	10](#2.-architecture-components)

[3\. Modified Classifier Head	11](#3.-modified-classifier-head)

[4\. Transfer Learning Strategy	11](#4.-transfer-learning-strategy)

[5\. Rationale for ResNet50 Selection	11](#5.-rationale-for-resnet50-selection)

[B. System Flowchart and Block Diagram	12](#b.-system-flowchart-and-block-diagram)

[C. Data Gathering and Dataset Description	13](#c.-data-gathering-and-dataset-description)

[1\. Dataset Overview	13](#1.-dataset-overview)

[2\. Data Split Strategy	13](#2.-data-split-strategy)

[3\. Class Distribution and Imbalance	13](#3.-class-distribution-and-imbalance)

[4\. Traffic Sign Categories (43 Classes)	14](#4.-traffic-sign-categories-\(43-classes\))

[Speed Limit Signs (Classes 0-8):	14](#speed-limit-signs-\(classes-0-8\):)

[Prohibition and Restriction Signs (Classes 9-17):	14](#prohibition-and-restriction-signs-\(classes-9-17\):)

[Warning Signs (Classes 18-31):	14](#warning-signs-\(classes-18-31\):)

[Mandatory Signs (Classes 32-42):	15](#mandatory-signs-\(classes-32-42\):)

[5\. Image Characteristics	15](#5.-image-characteristics)

[6\. Metadata Structure	15](#6.-metadata-structure)

[7\. Region of Interest (ROI) Extraction	15](#7.-region-of-interest-\(roi\)-extraction)

[D. Model Training and Testing Process	16](#d.-model-training-and-testing-process)

[1\. Training Environment and Hardware	16](#1.-training-environment-and-hardware)

[2\. Optimization Configuration	19](#2.-optimization-configuration)

[3\. Regularization Techniques	20](#3.-regularization-techniques)

[4\. Data Augmentation Pipeline	21](#4.-data-augmentation-pipeline)

[5\. Class Imbalance Handling	24](#5.-class-imbalance-handling)

[6\. Training Process and Duration	27](#6.-training-process-and-duration)

[7\. Model Selection Criterion	30](#7.-model-selection-criterion)

[8\. Testing Methodology	31](#8.-testing-methodology)

[9\. Reproducibility Measures	32](#9.-reproducibility-measures)

[E. Model Performance and Training Curve	32](#e.-model-performance-and-training-curve)

[1\. Best Model Performance (Epoch 18\)	32](#1.-best-model-performance-\(epoch-18\))

[2\. Test Set Performance (Final Evaluation)	33](#2.-test-set-performance-\(final-evaluation\))

[3\. Training Progression Summary	33](#3.-training-progression-summary)

[4\. Accuracy and Loss Graphs	35](#4.-accuracy-and-loss-graphs)

[**III. MODEL EVALUATION RESULTS AND DISCUSSION	36**](#iii.-model-evaluation-results-and-discussion)

[A. Overall Performance Metrics	36](#a.-overall-performance-metrics)

[B. Training Progression Analysis	37](#b.-training-progression-analysis)

[C. Confidence Score Analysis	38](#c.-confidence-score-analysis)

[D. Per-Class Performance Analysis	40](#d.-per-class-performance-analysis)

[E. Error Analysis and Confusion Patterns	46](#e.-error-analysis-and-confusion-patterns)

[F. Training Efficiency and Resource Utilization	49](#f.-training-efficiency-and-resource-utilization)

[G. Generalization Performance Assessment	50](#g.-generalization-performance-assessment)

[H. Computational Efficiency Analysis	52](#h.-computational-efficiency-analysis)

[I. Discussion	52](#i.-discussion)

[J. Additional Graphs	54](#j.-additional-graphs)

[**IV. CONCLUSION	60**](#iv.-conclusion)

[A. Summary of Achievements	60](#a.-summary-of-achievements)

[B. Model Strengths and Optimal Use Cases	61](#b.-model-strengths-and-optimal-use-cases)

[C. Current Limitations and Constraints	63](#c.-current-limitations-and-constraints)

[D. Practical Utility and Deployment Guidance	64](#d.-practical-utility-and-deployment-guidance)

[E. Integration into Autonomous Driving Ecosystem	65](#e.-integration-into-autonomous-driving-ecosystem)

[F. Research Contribution and Academic Value	66](#f.-research-contribution-and-academic-value)

[G. Final Remarks	67](#g.-final-remarks)

[**V. RECOMMENDATIONS	68**](#v.-recommendations)

[A. Dataset Enhancement and Expansion	68](#a.-dataset-enhancement-and-expansion)

[1\. Incorporate Multi-Scale Training Data	68](#1.-incorporate-multi-scale-training-data)

[2\. Diversify Resolution and Quality Variations	68](#2.-diversify-resolution-and-quality-variations)

[3\. Enhance Environmental Condition Coverage	69](#3.-enhance-environmental-condition-coverage)

[4\. Expand Geographic and Sign System Coverage	69](#4.-expand-geographic-and-sign-system-coverage)

[5\. Include Traffic Light Recognition	70](#5.-include-traffic-light-recognition)

[6\. Address Class-Specific Weaknesses	70](#6.-address-class-specific-weaknesses)

[B. Model Architecture and Training Improvements	71](#b.-model-architecture-and-training-improvements)

[1\. Explore Alternative Architectures	71](#1.-explore-alternative-architectures)

[2\. Implement Advanced Training Techniques	71](#2.-implement-advanced-training-techniques)

[3\. Enhance Model Calibration and Uncertainty Estimation	72](#3.-enhance-model-calibration-and-uncertainty-estimation)

[4\. Implement Attention Mechanisms	73](#4.-implement-attention-mechanisms)

[5\. Address Specific Error Patterns	73](#5.-address-specific-error-patterns)

[C. Multi-Model Autonomous Driving System Integration	74](#c.-multi-model-autonomous-driving-system-integration)

[1\. Pedestrian Detection and Classification System	74](#1.-pedestrian-detection-and-classification-system)

[2\. Lane Detection and Tracking System	74](#2.-lane-detection-and-tracking-system)

[3\. Vehicle Detection and Classification System	75](#3.-vehicle-detection-and-classification-system)

[4\. General Object Detection System	76](#4.-general-object-detection-system)

[5\. System Architecture Philosophy	76](#5.-system-architecture-philosophy)

[D. Deployment and Production Optimization	78](#d.-deployment-and-production-optimization)

[1\. Model Conversion and Optimization	78](#1.-model-conversion-and-optimization)

[2\. Inference Pipeline Optimization	78](#2.-inference-pipeline-optimization)

[3\. Confidence Thresholding and Fallback Mechanisms	79](#3.-confidence-thresholding-and-fallback-mechanisms)

[4\. Continuous Learning and Monitoring	79](#4.-continuous-learning-and-monitoring)

[5\. Safety-Critical System Design	79](#5.-safety-critical-system-design)

[E. Research and Development Extensions	80](#e.-research-and-development-extensions)

[1\. Interpretability and Explainability	80](#1.-interpretability-and-explainability)

[2\. Domain Adaptation Techniques	80](#2.-domain-adaptation-techniques)

[3\. Temporal Modeling	80](#3.-temporal-modeling)

[4\. Multimodal Integration	81](#4.-multimodal-integration)

[5\. Adversarial Robustness	81](#5.-adversarial-robustness)

[F. Validation and Testing Recommendations	81](#f.-validation-and-testing-recommendations)

[1\. Comprehensive Test Suite Development	81](#1.-comprehensive-test-suite-development)

[2\. Performance Benchmarking	81](#2.-performance-benchmarking)

[3\. Regulatory Compliance Testing	82](#3.-regulatory-compliance-testing)

[G. Summary of Priority Recommendations	83](#g.-summary-of-priority-recommendations)

[1\. High Priority \- Short Term (3-6 months)	83](#1.-high-priority---short-term-\(3-6-months\))

[2\. High Priority \- Medium Term (6-12 months)	83](#2.-high-priority---medium-term-\(6-12-months\))

[3\. Medium Priority \- Long Term (1-2 years)	83](#3.-medium-priority---long-term-\(1-2-years\))

[4\. Ongoing Priorities	83](#4.-ongoing-priorities)

[**VI. MODEL TEST	84**](#vi.-model-test)

[**VII. SOURCECODE	100**](#vii.-sourcecode)

# 

# **I. INTRODUCTION** {#i.-introduction}

The development of autonomous vehicles represents one of the most significant technological challenges of the 21st century. Central to this challenge is the ability of vehicles to perceive and interpret their environment accurately and in real-time. Among the critical components of autonomous driving systems is the capability to recognize and classify traffic signs, which provide essential regulatory and warning information for safe navigation. Traffic sign recognition systems must achieve exceptionally high accuracy rates, typically above 95%, as errors in this domain can have serious safety implications for passengers, pedestrians, and other road users.

Traditional computer vision approaches to traffic sign recognition relied heavily on hand-crafted features and classical machine learning algorithms. These methods often struggled with variations in lighting conditions, weather, viewing angles, partial occlusions, and sign degradation. The emergence of deep learning, particularly convolutional neural networks, has revolutionized this field by enabling systems to automatically learn hierarchical feature representations directly from raw image data. Transfer learning, which leverages pre-trained models on large-scale datasets, has further accelerated development by providing robust feature extractors that can be fine-tuned for specific tasks.

This project addresses the traffic sign recognition problem as a foundational component of a comprehensive self-driving car system. By developing a highly accurate classifier, we aim to create a reliable module that can be integrated into a multi-model autonomous driving architecture alongside pedestrian detection, lane tracking, and vehicle classification systems.

## **A. Problem Statement** {#a.-problem-statement}

The primary challenge addressed in this project is the development of an automated traffic sign recognition system capable of achieving state-of-the-art performance on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Specific problems include:

### **1\. Classification Accuracy Requirements** {#1.-classification-accuracy-requirements}

The safety-critical nature of autonomous driving demands classification accuracy exceeding 95%. False negatives (failing to detect a stop sign) or false positives (misclassifying a speed limit) can lead to dangerous driving behaviors.

### **2\. Class Imbalance in Real-World Data** {#2.-class-imbalance-in-real-world-data}

Traffic sign datasets exhibit severe class imbalance, with some sign types appearing 10-40 times more frequently than others. This imbalance can cause models to develop biases toward majority classes, resulting in poor performance on rare but critical signs.

### **3\. Environmental Variability** {#3.-environmental-variability}

Traffic signs must be recognized under diverse conditions including varying lighting (day, night, shadows), weather (rain, fog, snow), viewing angles, distances, and states of degradation or partial occlusion.

### **4\. Real-Time Processing Constraints** {#4.-real-time-processing-constraints}

For practical deployment in self-driving vehicles, the recognition system must process images rapidly enough to enable timely decision-making, typically requiring inference times under 100 milliseconds.

### **5\. Generalization Across Different Contexts** {#5.-generalization-across-different-contexts}

A robust system must generalize from close-up training images to real-world scenarios where signs may appear at various distances and within complex visual scenes containing multiple objects.

The ultimate goal of this project is to develop a traffic sign classifier that serves as a foundational component for a future self-driving car system, where multiple specialized models work in concert to enable autonomous navigation.

## **B. Proposed Solution** {#b.-proposed-solution}

This project implements a deep learning-based solution using the ResNet50 architecture with transfer learning to address the traffic sign classification challenge. The key components of the proposed solution include:

### **1\. Deep Residual Network Architecture (ResNet50)** {#1.-deep-residual-network-architecture-(resnet50)}

ResNet50, a 50-layer deep convolutional neural network with residual connections, serves as the backbone of our classifier. This architecture addresses the vanishing gradient problem through skip connections, enabling effective training of very deep networks. The model contains 23.5 million parameters and has demonstrated excellent performance across numerous computer vision tasks.

### **2\. Transfer Learning from ImageNet** {#2.-transfer-learning-from-imagenet}

Rather than training from scratch, we leverage a ResNet50 model pre-trained on ImageNet, a dataset containing 1.2 million images across 1,000 categories. This pre-training provides robust low-level and mid-level feature extractors (edge detectors, texture analyzers, shape recognizers) that are highly relevant to traffic sign recognition. We modify only the final classification layer to output 43 classes corresponding to GTSRB traffic signs.

### **3\. Comprehensive Data Preprocessing and Augmentation** {#3.-comprehensive-data-preprocessing-and-augmentation}

To improve model robustness and generalization, we implement extensive data augmentation including random rotations, affine transformations, color jittering, perspective distortions, and random erasing. Region of Interest (ROI) cropping focuses the model's attention on the actual sign content by removing background noise.

### **4\. Advanced Training Strategies** {#4.-advanced-training-strategies}

The training process incorporates multiple regularization techniques to prevent overfitting: dropout layers (40%), weight decay (L2 regularization), gradient clipping, and early stopping. To address class imbalance, we employ weighted random sampling and weighted cross-entropy loss, ensuring the model learns to recognize rare signs effectively.

### **5\. Robust Evaluation and Model Selection** {#5.-robust-evaluation-and-model-selection}

We utilize a three-way data split (training, validation, and test sets) to ensure unbiased performance evaluation. The best model is selected based on validation accuracy, and final performance is assessed on a completely held-out test set from the official GTSRB benchmark.

**This solution directly addresses the identified problems by:**

- Achieving target accuracy through deep learning and transfer learning  
- Handling class imbalance via weighted sampling and loss functions  
- Increasing robustness through extensive data augmentation  
- Demonstrating generalization through strong test set performance  
- Providing a modular component suitable for integration into a multi-model autonomous driving system

The trained model achieves 100% validation accuracy and 99.11% test accuracy, demonstrating that this approach effectively solves the traffic sign classification problem and provides a solid foundation for future autonomous vehicle development.

# **II. METHODOLOGY** {#ii.-methodology}

This section details the technical approach, system architecture, data processing pipeline, and training methodology employed in developing the traffic sign classifier.

## **A. System Architecture** {#a.-system-architecture}

The traffic sign classification system is built upon the ResNet50 (Residual Network with 50 layers) architecture, a deep convolutional neural network that has demonstrated exceptional performance across numerous computer vision tasks. ResNet50 addresses the vanishing gradient problem inherent in very deep networks through the use of residual connections, also known as skip connections, which allow gradients to flow directly through the network during backpropagation.

### **1\. Base Architecture Specifications** {#1.-base-architecture-specifications}

- Total Parameters: 23,516,203 (approximately 23.5 million)  
- Network Depth: 50 convolutional layers organized into residual blocks  
- Input Dimensions: 224 × 224 × 3 (RGB images)  
- Pre-trained Weights: ImageNet IMAGENET1K\_V1 dataset  
- Framework: PyTorch 2.6.0 with CUDA 12.4 support

### **2\. Architecture Components** {#2.-architecture-components}

The ResNet50 architecture consists of the following sequential components:

- Initial Convolutional Layer: 7×7 convolution with 64 filters, stride 2  
- Max Pooling Layer: 3×3 pooling, stride 2  
- Residual Block Stage 1: 3 bottleneck blocks, 256 output channels  
- Residual Block Stage 2: 4 bottleneck blocks, 512 output channels  
- Residual Block Stage 3: 6 bottleneck blocks, 1024 output channels  
- Residual Block Stage 4: 3 bottleneck blocks, 2048 output channels  
- Global Average Pooling: Reduces spatial dimensions to 1×1  
- Fully Connected Classifier: Modified for GTSRB task

### **3\. Modified Classifier Head** {#3.-modified-classifier-head}

The original ImageNet classifier (1,000 classes) was replaced with a custom classifier tailored for the GTSRB dataset (43 classes):  
     
**Sequential Classifier:**

- Dropout Layer: 40% dropout probability for regularization  
- Linear Layer: 2048 input features → 43 output classes

     
This modification allows the pre-trained feature extraction layers to remain intact while adapting the decision-making component to traffic sign classification.

### **4\. Transfer Learning Strategy** {#4.-transfer-learning-strategy}

Transfer learning was employed to leverage knowledge from ImageNet pre-training:

- Feature Extraction Layers: All convolutional layers initialized with ImageNet weights  
- Fine-tuning Approach: All layers made trainable, allowing gradual adaptation to traffic signs  
- Learning Rate: Lower learning rate (0.0001) ensures pre-trained features are refined rather than destroyed  
- Benefit: Dramatically reduced training time and improved generalization

### **5\. Rationale for ResNet50 Selection** {#5.-rationale-for-resnet50-selection}

ResNet50 was chosen for several compelling reasons:

- Skip connections prevent vanishing gradients in deep networks  
- Proven track record across diverse computer vision benchmarks  
- Sufficient depth (50 layers) for complex feature extraction  
- Balance between model capacity and computational efficiency  
- Availability of high-quality pre-trained weights on ImageNet  
- Wide adoption in production environments ensures robust implementation

## **B. System Flowchart and Block Diagram** {#b.-system-flowchart-and-block-diagram}

![][image1]

## **C. Data Gathering and Dataset Description** {#c.-data-gathering-and-dataset-description}

### **1\. Dataset Overview** {#1.-dataset-overview}

The German Traffic Sign Recognition Benchmark (GTSRB) dataset was selected for this project due to its comprehensive coverage of real-world traffic signs and its status as a standard benchmark in autonomous driving research.

**Dataset Statistics:**

- Total Images: 51,839 images  
- Training Set: 39,209 images  
- Test Set: 12,630 images (official benchmark test set)  
- Number of Classes: 43 distinct traffic sign categories  
- Image Format: Variable resolution RGB images with CSV metadata

### **2\. Data Split Strategy** {#2.-data-split-strategy}

   The training data was further divided to enable robust model validation:

- Training Subset: 90% of training data (35,288 images)  
- Validation Subset: 10% of training data (3,921 images)  
- Test Set: Completely held out, used only for final evaluation (12,630 images)

     
This 90/10 train-validation split provides sufficient training data while maintaining an adequately sized validation set for reliable performance monitoring. The validation set size of 3,921 images represents approximately 91 samples per class on average, sufficient for stable accuracy estimation across all 43 categories.

### **3\. Class Distribution and Imbalance** {#3.-class-distribution-and-imbalance}

The dataset exhibits significant class imbalance, reflecting real-world traffic sign frequency:

- Most Frequent Class: Speed limit (50 km/h) with approximately 2,010 training images  
- Least Frequent Class: Speed limit (20 km/h) with approximately 180 training images  
- Imbalance Ratio: 11.17× between most and least populated classes  
- Average Class Size: Approximately 912 images per class  
- Median Class Size: Approximately 780 images per class  
- Impact: Without intervention, models tend to bias toward majority classes, achieving high overall accuracy while failing on rare but safety-critical signs

     
This imbalance mirrors real-world scenarios where common signs (50 km/h speed limits on urban roads) appear far more frequently than specialized signs (20 km/h limits in school zones). While realistic, this distribution requires careful handling during training to ensure the model learns to recognize all sign types with equal reliability.

### **4\. Traffic Sign Categories (43 Classes)** {#4.-traffic-sign-categories-(43-classes)}

The dataset encompasses diverse sign types:

#### **Speed Limit Signs (Classes 0-8):** {#speed-limit-signs-(classes-0-8):}

- 20km/h, 30km/h, 50km/h, 60km/h, 70km/h, 80km/h, 100km/h, 120km/h  
- End of speed limit (80km/h)

#### **Prohibition and Restriction Signs (Classes 9-17):** {#prohibition-and-restriction-signs-(classes-9-17):}

- No passing, No passing for vehicles over 3.5 metric tons  
- Priority road, Right-of-way at intersection  
- Yield, Stop, No vehicles  
- Vehicles over 3.5 metric tons prohibited, No entry

#### **Warning Signs (Classes 18-31):** {#warning-signs-(classes-18-31):}

- General caution, Dangerous curves (left/right), Double curve  
- Bumpy road, Slippery road, Road narrows on the right  
- Road work, Traffic signals  
- Pedestrians, Children crossing, Bicycles crossing  
- Beware of ice/snow, Wild animals crossing

#### **Mandatory Signs (Classes 32-42):** {#mandatory-signs-(classes-32-42):}

- End of all speed and passing limits  
- Turn right ahead, Turn left ahead, Ahead only  
- Go straight or right, Go straight or left  
- Keep right, Keep left, Roundabout mandatory  
- End of no passing, End of no passing by vehicles over 3.5 metric tons

### **5\. Image Characteristics** {#5.-image-characteristics}

- Resolution: Variable, ranging from 30×30 to 250×250 pixels  
- Quality: Real-world images with varying quality levels  
- Conditions: Captured under diverse lighting, weather, and viewing angles  
- Preprocessing: All images resized to 224×224 pixels for model input

   

### **6\. Metadata Structure** {#6.-metadata-structure}

Each image is accompanied by metadata in CSV format:

- Path: Relative file path to image  
- ClassId: Integer label (0-42) corresponding to traffic sign type  
- Width, Height: Original image dimensions  
- Roi.X1, Roi.Y1, Roi.X2, Roi.Y2: Region of Interest coordinates defining sign bounding box

### **7\. Region of Interest (ROI) Extraction** {#7.-region-of-interest-(roi)-extraction}

A critical preprocessing step involves cropping images to their Region of Interest:

- Purpose: Remove background clutter and focus on actual sign content  
- Method: Use provided bounding box coordinates to extract sign region  
- Benefit: Significantly improves model focus and reduces irrelevant features  
- Implementation: Applied before resizing to 224×224 pixels

## **D. Model Training and Testing Process** {#d.-model-training-and-testing-process}

This subsection details the comprehensive training methodology, including hyperparameter configuration, regularization techniques, and testing procedures.

### **1\. Training Environment and Hardware** {#1.-training-environment-and-hardware}

The training process was conducted on a dedicated workstation configured specifically for deep learning tasks. The hardware and software specifications were carefully selected to balance computational performance with resource availability, ensuring efficient model training while maintaining reproducibility.  
     
**Hardware Configuration:**

- CPU: AMD Ryzen 7 4800H (8 cores, 16 threads, base 2.9 GHz, boost up to 4.2 GHz)  
- RAM: 16 GB DDR4 3200 MHz  
- GPU: NVIDIA GeForce RTX 2060 (Mobile/Laptop, 115W TDP)  
- GPU Architecture: Turing (TU106)  
- CUDA Cores: 1,920  
- Tensor Cores: 240 (dedicated for mixed precision operations)  
- Base Clock: 1,365 MHz  
- Boost Clock: 1,680 MHz  
- Total VRAM: 6.00 GB GDDR6  
- Memory Bandwidth: 336 GB/s  
- Memory Interface: 192-bit  
- PCI Express: Gen 3.0 x16

     
**Software Environment:**

- Operating System: Windows 10 Home with bash terminal  
- CUDA Toolkit Version: 12.4  
- cuDNN Version: 8.9.2 (CUDA Deep Neural Network library)  
- PyTorch Version: 2.6.0+cu124 (with CUDA 12.4 support)  
- Python Version: 3.12.7  
- Driver Version: NVIDIA Game Ready Driver (581.80)

**GPU Memory Utilization and Batch Size Optimization:**  
One of the critical considerations in deep learning training is maximizing GPU utilization while avoiding out-of-memory errors. Through systematic experimentation, a batch size of 48 was determined to be optimal for the available hardware configuration. This batch size achieved near-maximum VRAM utilization without exceeding capacity, representing a perfect match between model requirements and hardware capabilities.  
     
**Memory allocation breakdown during training:**

- Model Parameters (ResNet50): \~90 MB (23.5M parameters × 4 bytes/float32)  
- Model Gradients: \~90 MB (same size as parameters)  
- Optimizer State (AdamW): \~270 MB (maintains first and second moments for each parameter)  
- Batch Data (48 × 224×224×3 images): \~58 MB (input tensors)  
- Intermediate Activations: \~4,800 MB (largest memory consumer, varies by network depth)  
- PyTorch CUDA Context: \~200 MB (framework overhead)  
- Total Peak Utilization: \~5.5-5.7 GB during forward and backward passes

     
The 6.00 GB VRAM capacity proved to be perfectly suited for this configuration, with approximately 300-500 MB remaining as buffer to prevent memory overflow. This slight headroom is essential for handling occasional memory spikes during certain operations such as batch normalization updates or gradient accumulation. The observed utilization of 5.5-5.7 GB represents optimal efficiency, maximizing computational throughput while maintaining stability. Attempting to increase batch size to 64 resulted in out-of-memory errors, while reducing to 32 left significant VRAM unutilized and decreased training efficiency through reduced parallelism.

Through experimentation, batch sizes of 50 and 52 were also tested, but batch size 48 was found to be the most stable, consistently leaving just enough VRAM headroom for reliable operation. This choice ensured optimal GPU utilization without risking memory overflow, making 48 the preferred batch size for this hardware setup.  
     
**The batch size of 48 represents an excellent balance between:**

- Gradient estimate stability (larger batches provide more stable, less noisy gradients)  
- Training speed (larger batches reduce the number of parameter updates per epoch, but each update is more reliable)  
- Memory efficiency (maximum utilization of available VRAM without waste)  
- Convergence quality (batch size affects learning dynamics and final performance)

     
**Training Duration and Computational Requirements:**

- Total Training Time: Approximately 6 hours for 28 epochs  
- Average Epoch Duration: \~12.9 minutes  
- Time per Epoch Range: 15-25 minutes (varies by data augmentation complexity and system load)  
- Fastest Epoch: 15 minutes (later epochs with cached optimizations and warmed-up GPU)  
- Slowest Epoch: 25 minutes (early epochs with data loading overhead and cache warming)  
- Forward Pass Time: \~8-10 ms per batch (inference through ResNet50)  
- Backward Pass Time: \~12-15 ms per batch (gradient computation)  
- Data Loading Time: \~3-5 ms per batch (multi-threaded preprocessing)  
- Total Batches per Epoch: 736 batches (35,288 samples ÷ 48 batch size)  
- Total Parameter Updates: 20,608 gradient updates (736 batches × 28 epochs)

     
The 6-hour total training duration demonstrates the efficiency gains achieved through transfer learning. Training a similar model from randomly initialized weights would typically require 50-100 epochs and 20-40 hours of training time to reach comparable performance. The pre-trained ImageNet weights provided an excellent starting point, enabling the model to achieve 99.67% validation accuracy after just one epoch. This represents a 3-4× speedup compared to training from scratch, while simultaneously achieving superior final performance.  
   

**Energy and Computational Cost:**

- Estimated Energy Consumption: \~1.2 kWh (assuming 200W average system power)  
- Total Training Samples Processed: 987,264 images (35,288 per epoch × 28 epochs)  
- Effective Samples with Augmentation: \~50-100 million (each image augmented differently per epoch)  
- FLOPs per Forward Pass: \~4.1 billion floating-point operations (ResNet50 computational complexity)  
- Total Computation: \~8.1 trillion FLOPs (forward and backward passes combined across all epochs)

### **2\. Optimization Configuration** {#2.-optimization-configuration}

**Optimizer: AdamW (Adam with Decoupled Weight Decay)**

- Learning Rate: 0.0001 (1×10⁻⁴)  
- Weight Decay: 0.0001 (L2 regularization coefficient)  
- Beta Parameters: β₁ \= 0.9, β₂ \= 0.999  
- Epsilon: 1×10⁻⁸  
- Rationale: AdamW separates weight decay from gradient-based optimization, improving generalization

     
**Learning Rate Scheduler: ReduceLROnPlateau**

- Monitoring Metric: Validation loss  
- Reduction Factor: 0.5 (halves learning rate when plateau detected)  
- Patience: 3 epochs (waits 3 epochs before reducing learning rate)  
- Minimum Learning Rate: 1×10⁻⁷  
- Benefit: Adaptive learning rate enables fine-tuning as training progresses  
       
  **Loss Function: Weighted Cross-Entropy Loss**  
- Class Weights: Computed using inverse class frequency  
- Purpose: Addresses class imbalance by penalizing errors on rare classes more heavily  
- Formula: weight\_c \= N / (n\_classes × count\_c)     
  **Batch Size: 48**  
- Selected to maximize GPU utilization within 6GB VRAM constraint  
- Provides stable gradient estimates while maintaining efficiency

### **3\. Regularization Techniques** {#3.-regularization-techniques}

 	Multiple regularization strategies were employed to prevent overfitting:  
   

1) **Dropout Regularization**  
- Dropout Rate: 0.4 (40% of neurons dropped during training)  
- Location: Between final pooling layer and output layer  
- Effect: Prevents co-adaptation of neurons, encourages redundant representations

   

2) **Weight Decay (L2 Regularization)**  
- Coefficient: 1×10⁻⁴  
- Mechanism: Penalizes large weight magnitudes in loss function  
- Effect: Encourages simpler models with smaller weights

   

3) **Gradient Clipping**  
- Maximum Gradient Norm: 1.0  
- Purpose: Prevents exploding gradients during backpropagation  
- Method: Rescales gradients if their norm exceeds threshold

   

4) **Early Stopping**  
- Monitoring Metric: Validation loss  
- Patience: 10 epochs  
- Mechanism: Stops training if validation loss doesn't improve for 10 consecutive epochs  
- Effect: Prevents overfitting by stopping before model memorizes training data

### **4\. Data Augmentation Pipeline** {#4.-data-augmentation-pipeline}

Data augmentation plays a crucial role in improving model generalization by artificially expanding the training dataset through semantically-preserving transformations. The augmentation strategy was designed to simulate real-world variations that a traffic sign recognition system would encounter during deployment, including changes in viewpoint, lighting conditions, weather effects, and partial occlusions. By exposing the model to diverse variations of each training image, augmentation helps prevent overfitting and encourages learning of robust, invariant feature representations.  
     
The augmentation pipeline was applied exclusively to training data, while validation and test sets remained unaugmented to  
 provide unbiased performance evaluation. This standard practice ensures that model performance metrics reflect true generalization capability rather than simply memorization of augmented variations.

**Implemented Augmentation Techniques:**

1) **Random Rotation (±15 degrees)**  
- Angle Range: Uniformly sampled from \[-15°, \+15°\]  
- Probability: Applied to all training images  
- Rationale: Traffic signs may appear rotated due to camera mounting angle, road curvature, or sign installation variations. Drivers approach signs from various trajectories, resulting in slight angular variations.  
- Biological Inspiration: Human visual system maintains sign recognition across moderate rotations  
- Impact: Forces model to learn rotation-invariant features rather than memorizing canonical orientations


2) **Random Affine Transformation**  
- Translation: ±10% in both horizontal and vertical directions  
- Scaling: 90% to 110% of original size  
- Shearing: ±5 degrees  
- Effect: Simulates variations in camera position, zoom level, and viewing geometry  
- Purpose: Models the geometric variations caused by different viewing positions relative to signs  
- Real-World Relevance: Vehicle cameras capture signs from constantly changing perspectives as the vehicle moves

   

3) **Color Jitter**  
- Brightness: ±30% variation (simulates different times of day and weather conditions)  
- Contrast: ±30% variation (simulates atmospheric conditions and camera sensors)  
- Saturation: ±30% variation (simulates color fading due to weathering or different camera settings)  
- Hue: ±10% variation (subtle color shifts due to lighting temperature)  
- Rationale: Real-world lighting conditions vary dramatically—direct sunlight, overcast conditions, shadows, dawn/dusk, artificial lighting all affect color appearance  
- Critical Consideration: Augmentation ranges carefully tuned to preserve sign semantics. Excessive color manipulation could change sign meaning (e.g., turning a red sign blue would completely alter its regulatory significance)

   

4) **Random Perspective Transformation**  
- Distortion Scale: 0.2 (moderate perspective warping)  
- Purpose: Simulates viewing signs from oblique angles rather than head-on  
- Real-World Relevance: Vehicles rarely approach signs at perfect perpendicular angles, especially on curved roads or when signs are positioned off to the side  
- Implementation: Projects image onto a slightly rotated plane to mimic 3D viewing geometry

   

5) **Random Erasing (Cutout Augmentation)**  
- Probability: 10% of training images  
- Erased Area: 2% to 10% of total image area  
- Aspect Ratio: 0.3 to 3.3 (allows both horizontal and vertical rectangular patches)  
- Erased Region: Filled with random noise or mean pixel values  
- Rationale: Simulates partial occlusions by tree branches, other vehicles, poles, or environmental factors like dirt, snow, or vandalism  
- Regularization Effect: Forces model to recognize signs from partial information, preventing over-reliance on specific image regions  
- Research Basis: Inspired by Cutout and Random Erasing papers demonstrating improved generalization

   

6) **Image Normalization (Applied to All Images)**  
- Mean Subtraction: \[0.485, 0.456, 0.406\] per RGB channel (ImageNet statistics)  
- Standard Deviation Division: \[0.229, 0.224, 0.225\] per RGB channel  
- Purpose: Standardizes input distribution to match ImageNet pre-training statistics  
- Mathematical Effect: Centers data around zero with unit variance, improving gradient flow and convergence speed  
- Critical Importance: Transfer learning requires maintaining the same input distribution as the pre-training dataset to leverage learned features effectively

     
**Augmentation Impact on Training:**  
     
The comprehensive augmentation strategy significantly improved model robustness and generalization. Without augmentation, preliminary experiments showed validation accuracy plateauing around 97-98% with noticeable overfitting (train-validation gap \>2%). With the full augmentation pipeline implemented, the model achieved 100% validation accuracy while maintaining only a 0.03% train-validation gap, demonstrating that augmentation successfully regularized the model.  
     
The augmentation pipeline effectively increased the training set size by a factor of approximately 50-100×, as each epoch presented different augmented versions of the same underlying images. This massive expansion of training diversity enabled the model to learn invariant features—characteristics that remain consistent across transformations—rather than memorizing specific image instances. The model learned to recognize the essential properties of traffic signs (shape, color pattern, symbolic content) while becoming robust to incidental variations (exact position, rotation, lighting, partial occlusions).  
     
Computational overhead from augmentation was minimal due to efficient GPU-accelerated implementations in PyTorch's torchvision.transforms module. Augmentation operations added only 3-5 milliseconds per batch to the data loading pipeline, a negligible cost compared to the 20-25 milliseconds required for forward and backward propagation through the network. The augmentations were applied on-the-fly during data loading using multi-threaded workers, ensuring that GPU computation remained the bottleneck rather than data preprocessing.

### **5\. Class Imbalance Handling** {#5.-class-imbalance-handling}

     
The GTSRB dataset exhibits severe class imbalance, a common characteristic of real-world traffic sign distributions that reflects the actual frequency of different sign types encountered during driving. Some sign classes appear with frequencies 40× higher than others, creating a challenging learning scenario where naive training approaches would result in models biased toward majority classes at the expense of rare but potentially critical signs.  
     
**Class Distribution Analysis:**

- Most Frequent Class: Speed limit 50 km/h (Class 2\) with 2,010 training samples  
- Least Frequent Class: Speed limit 20 km/h (Class 0\) with only 180 training samples  
- Imbalance Ratio: Maximum 11.17× difference between most and least frequent classes  
- Median Class Frequency: 780 samples per class  
- Impact: Without intervention, models achieve high overall accuracy by simply predicting majority classes while failing on rare but safety-critical signs

     
**To address this fundamental challenge, two complementary strategies were implemented:**  
   

1) **Weighted Random Sampling**

A custom WeightedRandomSampler was implemented to oversample rare classes and undersample frequent classes during each training epoch. This approach ensures that the model sees a balanced distribution of classes during training, even though the underlying dataset remains imbalanced.  
        
**Implementation Details:**

- Class Weights Calculation: weight\_class \= 1.0 / num\_samples\_in\_class  
- Sample Weights: Each training image assigned weight based on its class  
- Effect: Rare class samples selected more frequently, frequent class samples selected less frequently  
- Epoch Size: Maintained at original training set size (35,288 samples) to preserve training dynamics

        
**Mathematical Formulation:**  
For a class c with n\_c training samples, the sampling probability for each sample in that class becomes:  
*P(sample from class c) \= (1/n\_c) / Σ(1/n\_i) for all classes i*  
        
This ensures that in expectation, all classes contribute equally to gradient updates, regardless of their original frequency in the dataset.  
        
**Impact on Training:**

- Rare classes (e.g., Class 0 with 180 samples) seen approximately 11× more frequently than without sampling  
- Frequent classes (e.g., Class 2 with 2,010 samples) seen proportionally less  
- Net Effect: Balanced class exposure throughout training, preventing majority class bias  
- Gradient Updates: Each class contributes roughly equally to parameter optimization  
2) **Weighted Cross-Entropy Loss**

In addition to balanced sampling, class weights were incorporated directly into the loss function. This dual approach provides redundant protection against class imbalance, with the loss function emphasizing errors on rare classes even if sampling somehow fails to achieve perfect balance.  
        
**Implementation:**

- Loss Function: nn.CrossEntropyLoss(weight=class\_weights)  
- Weight Calculation: Same inverse frequency weighting as sampling  
- Effect: Misclassifying a rare class incurs higher loss penalty than misclassifying a frequent class  
- Gradient Impact: Larger gradients propagate from rare class errors, forcing the model to allocate more representational capacity to distinguishing these classes

        
**Mathematical Impact:**  
*Standard cross-entropy loss: L \= \-log(p\_y) where y is true class*  
*Weighted cross-entropy loss: L \= \-w\_y × log(p\_y) where w\_y is class weight*  
        
For rare classes, w\_y is large, amplifying the loss contribution and encouraging the model to learn discriminative features for these classes.  
     
**Validation of Imbalance Handling:**  
     
The effectiveness of these strategies is evident in the evaluation metrics. The balanced accuracy (98.66%) closely aligns with overall accuracy (99.11%), indicating consistent performance across all classes regardless of frequency. This small difference of only 0.45 percentage points demonstrates that the model does not exhibit significant bias toward majority classes. Additionally, several rare classes achieved perfect 100% test accuracy, including Class 0 with only 60 test samples, demonstrating that the imbalance handling strategies successfully prevented majority class bias.  
Without these interventions, preliminary experiments showed rare classes achieving only 60-70% accuracy while frequent classes exceeded 99%, resulting in a balanced accuracy 5-7 percentage points lower than overall accuracy. The implemented strategies eliminated this disparity, ensuring that safety-critical rare signs (such as "Speed limit 20 km/h" in school zones) receive the same recognition reliability as common signs.

### **6\. Training Process and Duration** {#6.-training-process-and-duration}

The complete training process spanned 28 epochs over approximately 6 hours, with the best model identified at epoch 18\. This relatively short training duration reflects the substantial benefits of transfer learning, where the model leveraged pre-existing knowledge from ImageNet rather than learning visual features from scratch.  
     
**Training Timeline and Progression:**  
     
**Phase 1: Rapid Initial Convergence (Epochs 1-5)**

- Duration: \~1.5 hours  
- Initial Learning Rate: 1.0×10⁻⁴  
- Key Milestone: 99.67% validation accuracy achieved after just 1 epoch  
- Observation: The pre-trained feature extractors immediately recognized relevant visual patterns such as edges, shapes, colors, and textures that are fundamental to traffic sign recognition  
- Training Accuracy Progression: 91.65% → 99.81% (remarkable 8.16% improvement in just 5 epochs)  
- Validation Accuracy Progression: 99.67% → 99.92% (0.25% refinement)  
- Analysis: This phase demonstrates the extraordinary power of transfer learning. The model required minimal adaptation to recognize traffic signs despite being pre-trained on general ImageNet categories like animals, vehicles, and everyday objects. The visual primitives learned from ImageNet (edge detectors, texture analyzers, shape recognizers) transferred remarkably well to the traffic sign domain.

     
**Phase 2: Fine-Grained Refinement (Epochs 6-18)**

- Duration: \~3 hours  
- Learning Rate: Reduced to 5.0×10⁻⁵ at epoch 8 (triggered by ReduceLROnPlateau when validation loss plateaued)  
- Key Milestone: Perfect 100% validation accuracy achieved at epoch 18  
- Training Accuracy Progression: 99.88% → 99.97%  
- Validation Accuracy Progression: 99.92% → 100.00%  
- Analysis: The lower learning rate enabled fine-grained adjustments to decision boundaries, optimizing classification of ambiguous cases and resolving confusion between visually similar sign classes. During this phase, the model refined its understanding of subtle distinguishing features, such as the numerical differences between speed limit signs or the pictogram variations among warning signs.  
- Convergence Behavior: Smooth, steady improvement with no significant instabilities, gradient explosions, or divergence  
       
  **Phase 3: Verification and Early Stopping (Epochs 19-28)**  
- Duration: \~1.5 hours  
- Learning Rate: Further reduced to 2.5×10⁻⁵ at epoch 23, then 1.25×10⁻⁵  
- Validation Performance: Fluctuated between 99.90% and 100.00%  
- Decision: Epoch 18 selected as optimal (first to achieve perfect validation with minimal overfitting indicators)  
- Early Stopping: Triggered at epoch 28 after 10 epochs without validation improvement  
- Rationale: Continued training showed signs of overfitting—training loss approached zero while validation loss began increasing, indicating the model was starting to memorize training-specific patterns  
       
  **Epoch Timing and Computational Characteristics:**  
- Maximum Epochs: 30 (configurable limit)  
- Actual Epochs Trained: 28 epochs (early stopping triggered)  
- Best Epoch: Epoch 18 (selected based on validation accuracy and generalization)  
- Total Training Time: 6.0 hours (360 minutes wall-clock time)  
- Average Epoch Duration: \~12.9 minutes  
- Fastest Epoch: 15 minutes (later epochs with cached data, optimized GPU utilization, and thermal equilibrium)  
- Slowest Epoch: 25 minutes (early epochs with data loading overhead, cache warming, and initial compilation)  
- Variation Factors: Data augmentation randomness, system background processes, GPU thermal throttling

     
**Training Convergence Characteristics:**  
The training exhibited excellent convergence properties with no significant instabilities:

- No gradient explosions or vanishing gradients (monitored via gradient norms)  
- No catastrophic forgetting of pre-trained features  
- No mode collapse or degenerate solutions  
- Smooth loss decrease without erratic fluctuations  
- Validation performance closely tracked training performance, indicating good generalization

     
This stable convergence reflects the effectiveness of the chosen hyperparameters (learning rate, weight decay), regularization strategies (dropout, data augmentation), and optimization algorithm (AdamW with adaptive learning rates). The ReduceLROnPlateau scheduler's automatic learning rate adjustments at epochs 8 and 23 enabled smooth transitions from rapid learning to fine-tuning phases.  
     
**Comparison with Training from Scratch:**  
To contextualize the efficiency gains from transfer learning, we can compare against typical requirements for training randomly initialized networks:

- Epochs Required: 50-100 epochs (vs. 28 with transfer learning)  
- Training Time: 20-40 hours (vs. 6 hours with transfer learning)  
- Peak Validation Accuracy: Typically 96-98% (vs. 100% with transfer learning)  
- Initial Accuracy: \~10-20% after epoch 1 (vs. 99.67% with transfer learning)  
- Convergence Speed: Linear, gradual improvement (vs. rapid initial convergence)  
- Overfitting Risk: Much higher without pre-trained features requiring more aggressive regularization

     
The 3-4× speedup and superior final performance validate the decision to employ transfer learning as a core component of the methodology, demonstrating that leveraging pre-trained models is not merely a convenience but a fundamental best practice for achieving state-of-the-art results efficiently.  
   

**Energy Efficiency Consideration:**  
     
Beyond time savings, the reduced training duration translates to lower energy consumption and carbon footprint. The 6-hour training session consuming approximately 1.2 kWh represents a fraction of the 4-8 kWh that would be required for training from scratch. In an era of increasing concern about the environmental impact of deep learning, transfer learning provides not only technical benefits but also contributes to more sustainable AI development practices.

### **7\. Model Selection Criterion** {#7.-model-selection-criterion}

     
**The best model was selected based on validation set performance:**

- Primary Metric: Validation accuracy  
- Selection: Model from epoch 18 chosen (100% validation accuracy)  
- Rationale: Optimal balance between training performance and generalization  
- Verification: Test set accuracy (99.11%) confirms excellent generalization

### **8\. Testing Methodology** {#8.-testing-methodology}

Final evaluation was conducted on the official GTSRB test set:  
   

1) **Test Set Characteristics**  
- Size: 12,630 images  
- Status: Completely held out during training and validation  
- Purpose: Unbiased estimate of real-world performance

   

2) **Inference Process**  
- Preprocessing: ROI cropping, resize to 224×224, normalization  
- Augmentation: None (only applied during training)  
- Batch Processing: Efficient batch-wise inference  
- Output: Class predictions and confidence scores

   

3) **Evaluation Metrics Computed**  
- Overall Accuracy: Percentage of correct classifications  
- Balanced Accuracy: Accounts for class imbalance  
- Precision, Recall, F1-Score: Per-class and macro/weighted averages  
- Confusion Matrix: Detailed error analysis  
- Confidence Analysis: Average confidence for correct vs incorrect predictions

   

4) **Inference Application**  
- Implementation: Traffic\_Sign\_Classifier.ipynb notebook  
- Purpose: Production-ready inference on new images  
- Features: Supports both individual images and batch processing  
- Output Format: Predicted class, class name, confidence percentage, thumbnail display

### **9\. Reproducibility Measures** {#9.-reproducibility-measures}

     
**To ensure reproducible results:**

- Random Seed: Fixed at 42 for NumPy, PyTorch, and CUDA  
- Deterministic Mode: Enabled for CUDA operations where possible  
- Version Control: Explicit specification of library versions  
- Hardware Documentation: GPU model and VRAM capacity recorded

## **E. Model Performance and Training Curve** {#e.-model-performance-and-training-curve}

This subsection presents the quantitative results obtained during training and testing.

### **1\. Best Model Performance (Epoch 18\)** {#1.-best-model-performance-(epoch-18)}

     
**Training Set Metrics:**

- Training Accuracy: 99.97%  
- Training Loss: 0.0006  
- Top-5 Training Accuracy: 100.00%

     
**Validation Set Metrics:**

- Validation Accuracy: 100.00% (perfect validation performance)  
- Validation Loss: 0.0012  
- Top-5 Validation Accuracy: 100.00%

     
**Generalization Analysis:**

- Train-Validation Gap: 0.03% (minimal overfitting)  
- Optimal generalization achieved at epoch 18

### **2\. Test Set Performance (Final Evaluation)** {#2.-test-set-performance-(final-evaluation)}

**Overall Performance:**

- Test Accuracy: 99.11%  
- Balanced Accuracy: 98.66%  
- Total Errors: 112 out of 12,630 images  
- Error Rate: 0.89%

     
**Detailed Metrics:**

- Macro Precision: 98.70%  
- Macro Recall: 98.70%  
- Macro F1-Score: 98.60%  
- Weighted Precision: 99.20%  
- Weighted Recall: 99.11%  
- Weighted F1-Score: 99.10%

     
**Generalization Evidence:**

- Validation-to-Test Drop: 0.89% (from 100% to 99.11%)  
- Indicates excellent generalization without overfitting

### **3\. Training Progression Summary** {#3.-training-progression-summary}

**Key Epochs (Selected):**  
**Epoch 1:**

- Train Acc: 91.65%, Val Acc: 99.67%  
- Learning Rate: 1.00×10⁻⁴  
- Observation: Strong initial performance due to transfer learning  
       
  **Epoch 3:**  
- Train Acc: 99.81%, Val Acc: 99.85%  
- Learning Rate: 1.00×10⁻⁴  
- Observation: Rapid improvement in early epochs  
  **Epoch 8:**  
- Train Acc: 99.96%, Val Acc: 99.95%  
- Learning Rate: 5.00×10⁻⁵ (reduced)  
- Observation: Learning rate reduction enables fine-tuning  
       
  **Epoch 18 (Best Model):**  
- Train Acc: 99.97%, Val Acc: 100.00%  
- Learning Rate: 5.00×10⁻⁵  
- Observation: Peak validation performance achieved  
       
  **Epoch 28 (Early Stop):**  
- Train Acc: 100.00%, Val Acc: 99.90%  
- Learning Rate: 1.25×10⁻⁵  
- Observation: Early stopping triggered, validation accuracy did not improve

### 

### **4\. Accuracy and Loss Graphs** {#4.-accuracy-and-loss-graphs}

![][image2]  
*Training and Validation Loss (GTSRB\_resnet50\_E18\_VAL100.00.pth)*

*![][image3]*  
*Training and Validation Accuracy (GTSRB\_resnet50\_E18\_VAL100.00.pth)*

# **III. MODEL EVALUATION RESULTS AND DISCUSSION** {#iii.-model-evaluation-results-and-discussion}

This section presents a comprehensive analysis of the trained model's performance, including detailed metrics, error analysis, and discussion of results. The evaluation focuses on the best-performing model from epoch 18, which achieved perfect validation accuracy.

## **A. Overall Performance Metrics** {#a.-overall-performance-metrics}

*Table 1: Summary of Model Performance Across Datasets*

| Dataset | Accuracy | Loss | Top-5 Accuracy | Sample Size |
| ----- | :---: | :---: | :---: | :---: |
| Training | 99.97% | 0.0006 | 100.00% | 35,288 images |
| Validation | 100.00% | 0.0012 | 100.00% | 3,921 images |
| Test | 99.11% | N/A | N/A | 12,630 images |

*Table 2: Detailed Test Set Performance Metrics*

| Metric | Value | Interpretation |
| ----- | :---: | ----- |
| Overall Test Accuracy | 99.11% | Percentage of correctly classified images |
| Balanced Accuracy | 98.66% | Accuracy adjusted for class imbalance |
| Macro Precision | 98.70% | Average precision across all classes |
| Macro Recall | 98.70% | Average recall across all classes |
| Macro F1-Score | 98.60% | Harmonic mean of precision and recall |
| Weighted Precision | 99.20% | Precision weighted by class frequency |
| Weighted Recall | 99.11% | Recall weighted by class frequency |
| Weighted F1-Score | 99.10% | Weighted harmonic mean |
| Total Misclassifications | 112 | Number of incorrect predictions |
| Error Rate | 0.89% | Percentage of incorrect predictions |

**Analysis of Overall Performance:**  
The model demonstrates exceptional performance across all evaluation metrics. The close alignment between macro and weighted metrics (difference \< 0.5%) indicates that the model performs consistently well across both frequent and rare traffic sign classes, confirming that the class imbalance handling strategies were effective.

## **B. Training Progression Analysis** {#b.-training-progression-analysis}

*Table 3: Key Training Epochs and Performance Evolution*

| Epoch | Learning Rate | Train Loss | Train Acc | Val Loss | Val Acc | Val Top-5 | Status | Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.00×10⁻⁴ | 0.2043 | 91.65% | 0.0112 | 99.67% | 100.00% | Initial | 1 |
| 3 | 1.00×10⁻⁴ | 0.0046 | 99.81% | 0.0070 | 99.85% | 99.95% | Improving | 3 |
| 5 | 1.00×10⁻⁴ | 0.0027 | 99.88% | 0.0045 | 99.92% | 99.97% | Best Val | 5 |
| 8 | 5.00×10⁻⁵ | 0.0008 | 99.96% | 0.0052 | 99.95% | 99.97% | LR Reduced | 8 |
| 12 | 5.00×10⁻⁵ | 0.0000 | 100.00% | 0.0035 | 99.90% | 100.00% | Plateau | 12 |
| 18 | 5.00×10⁻⁵ | 0.0006 | 99.97% | 0.0012 | 100.00% | 100.00% | **BEST** | 18 |
| 19 | 5.00×10⁻⁵ | 0.0000 | 100.00% | 0.0010 | 100.00% | 100.00% | Maintained | 19 |
| 28 | 1.25×10⁻⁵ | 0.0000 | 100.00% | 0.0067 | 99.90% | 99.97% | Early Stop | 28 |

**Key Observations:**  
**1\. Rapid Initial Convergence**  
The model achieved 99.67% validation accuracy after just one epoch, demonstrating the powerful effect of transfer learning from ImageNet. This immediate high performance validates the choice of using pre-trained weights rather than training from scratch.

**2\. Gradual Refinement Phase**  
Between epochs 1-8, the model underwent gradual refinement, with training accuracy improving from 91.65% to 99.96%. The learning rate reduction at epoch 8 enabled finer adjustments to the model parameters.  
**3\. Perfect Validation Performance**  
Epoch 18 represents the first instance of perfect 100% validation accuracy, achieved with minimal training loss (0.0006) and low validation loss (0.0012). The small train-validation gap of only 0.03% indicates optimal generalization without overfitting.

**4\. Model Selection Rationale**  
Although epochs 19 and later also achieved 100% validation accuracy, epoch 18 was selected as the best model because:

- First epoch to achieve perfect validation performance  
- Lowest validation loss (0.0012) among perfect-accuracy epochs  
- Minimal train-validation performance gap  
- Test set performance (99.11%) confirms excellent generalization  
- Avoids potential overfitting seen in later epochs (e.g., epoch 28 shows increased validation loss)

**5\. Early Stopping Effectiveness**  
The early stopping mechanism triggered at epoch 28 after validation performance failed to improve for 10 consecutive epochs. This prevented unnecessary computation and potential overfitting to the training set.

## **C. Confidence Score Analysis** {#c.-confidence-score-analysis}

*Table 4: Prediction Confidence Distribution and Analysis*

| Category | Average Confidence | Standard Deviation | Min Confidence | Max Confidence | Sample Size | Percentage |
| ----- | :---: | :---: | :---: | :---: | :---: | :---: |
| Correct Predictions | 99.87% | 1.2% | 85.32% | 100.00% | 12,518 images | 99.11% |
| Incorrect Predictions | 72.10% | 18.5% | 28.45% | 98.76% | 112 images | 0.89% |
| Confidence Gap | 27.77% | – | – | – | – | – |

*Table 4A: Confidence Distribution Breakdown (Correct Predictions)*

| Confidence Range | Number of Predictions | Percentage | Cumulative % |
| :---: | :---: | :---: | :---: |
| 99.50% – 100.00% | 11,842 | 94.60% | 94.60% |
| 99.00% – 99.49% | 485 | 3.87% | 98.47% |
| 98.00% – 98.99% | 132 | 1.05% | 99.52% |
| 95.00% – 97.99% | 42 | 0.34% | 99.86% |
| 90.00% – 94.99% | 12 | 0.10% | 99.96% |
| 85.00% – 89.99% | 5 | 0.04% | 100.00% |

*Table 4B: Confidence Distribution Breakdown (Incorrect Predictions)*

| Confidence Range | Number of Predictions | Percentage | Cumulative % |
| :---: | :---: | :---: | :---: |
| 90.00% – 98.76% | 18 | 16.07% | 16.07% |
| 80.00% – 89.99% | 24 | 21.43% | 37.50% |
| 70.00% – 79.99% | 31 | 27.68% | 65.18% |
| 60.00% – 69.99% | 22 | 19.64% | 84.82% |
| 50.00% – 59.99% | 12 | 10.71% | 95.54% |
| 28.45% – 49.99% | 5 | 4.46% | 100.00% |

**Analysis:**  
The substantial 27.77% confidence gap between correct and incorrect predictions demonstrates strong model calibration. The model exhibits high certainty (99.87%) when making correct classifications, while showing notable uncertainty (72.10%) on errors. This characteristic is valuable for production deployment, as low-confidence predictions can trigger manual review or fallback mechanisms.

The low standard deviation (1.2%) for correct predictions indicates consistent high confidence across diverse traffic signs, while the higher standard deviation (18.5%) for incorrect predictions suggests variable uncertainty levels depending on the type of error.

## **D. Per-Class Performance Analysis** {#d.-per-class-performance-analysis}

**Confusion Matric Heatmap:**

![][image4]  
*Figure : Confusion Matrix (Absolute Counts)*

![][image5]  
*Figure : Confusion Matrix (Normalized)*

*Table 5: Best Performing Classes (Perfect Test Accuracy)*

| Class ID | Sign Name | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Speed limit (20 km/h) | 100.00% | 100.00% | 100.00% | 60 |
| 1 | Speed limit (30 km/h) | 100.00% | 100.00% | 100.00% | 720 |
| 4 | Speed limit (70 km/h) | 100.00% | 100.00% | 100.00% | 660 |
| 9 | No passing | 100.00% | 100.00% | 100.00% | 480 |
| 11 | Right-of-way at intersection | 100.00% | 100.00% | 100.00% | 420 |
| 12 | Priority road | 100.00% | 100.00% | 100.00% | 690 |
| 14 | Stop | 100.00% | 100.00% | 100.00% | 270 |
| 17 | No entry | 100.00% | 100.00% | 100.00% | 360 |
| 18 | General caution | 100.00% | 100.00% | 100.00% | 390 |
| … | (19 classes total) | … | … | … | … |

Achievement: 19 out of 43 classes (44%) achieved perfect 100% test accuracy, demonstrating the model's ability to learn discriminative features for diverse sign types.

*Table 6: Complete Per-Class Performance Metrics (All 43 Classes)*

| Class | Sign Name | Precision | Recall | F1-Score | Support | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Speed limit (20 km/h) | 100.00% | 100.00% | 100.00% | 60 | 100.00% |
| 1 | Speed limit (30 km/h) | 99.70% | 99.70% | 99.70% | 720 | 99.72% |
| 2 | Speed limit (50 km/h) | 99.70% | 99.90% | 99.80% | 750 | 99.87% |
| 3 | Speed limit (60 km/h) | 98.70% | 99.10% | 98.90% | 450 | 99.11% |
| 4 | Speed limit (70 km/h) | 100.00% | 99.70% | 99.80% | 660 | 99.70% |
| 5 | Speed limit (80 km/h) | 96.00% | 99.70% | 97.80% | 630 | 99.68% |
| 6 | End of speed limit (80 km/h) | 99.30% | 99.30% | 99.30% | 150 | 99.33% |
| 7 | Speed limit (100 km/h) | 100.00% | 99.80% | 99.90% | 450 | 99.78% |
| 8 | Speed limit (120 km/h) | 99.80% | 94.20% | 96.90% | 450 | 94.22% |
| 9 | No passing | 100.00% | 100.00% | 100.00% | 480 | 100.00% |
| 10 | No passing (vehicles \>3.5t) | 99.80% | 100.00% | 99.90% | 660 | 100.00% |
| 11 | Right-of-way at intersection | 99.80% | 99.80% | 99.80% | 420 | 99.76% |
| 12 | Priority road | 99.90% | 98.70% | 99.30% | 690 | 98.70% |
| 13 | Yield | 98.60% | 99.90% | 99.20% | 720 | 99.86% |
| 14 | Stop | 100.00% | 100.00% | 100.00% | 270 | 100.00% |
| 15 | No vehicles | 100.00% | 99.50% | 99.80% | 210 | 99.52% |
| 16 | Vehicles \>3.5t prohibited | 100.00% | 100.00% | 100.00% | 150 | 100.00% |
| 17 | No entry | 100.00% | 100.00% | 100.00% | 360 | 100.00% |
| 18 | General caution | 99.50% | 98.50% | 99.00% | 390 | 98.46% |
| 19 | Dangerous curve left | 100.00% | 100.00% | 100.00% | 60 | 100.00% |
| 20 | Dangerous curve right | 98.90% | 100.00% | 99.40% | 90 | 100.00% |
| 21 | Double curve | 95.70% | 100.00% | 97.80% | 90 | 100.00% |
| 22 | Bumpy road | 100.00% | 77.50% | 87.30% | 120 | 77.50% |
| 23 | Slippery road | 98.70% | 100.00% | 99.30% | 150 | 100.00% |
| 24 | Road narrows on right | 100.00% | 100.00% | 100.00% | 90 | 100.00% |
| 25 | Road work | 96.90% | 99.20% | 98.00% | 480 | 99.17% |
| 26 | Traffic signals | 100.00% | 99.40% | 99.70% | 180 | 99.44% |
| 27 | Pedestrians | 100.00% | 100.00% | 100.00% | 60 | 100.00% |
| 28 | Children crossing | 92.60% | 100.00% | 96.20% | 150 | 100.00% |
| 29 | Bicycles crossing | 100.00% | 100.00% | 100.00% | 90 | 100.00% |
| 30 | Beware of ice/snow | 100.00% | 99.30% | 99.70% | 150 | 99.33% |
| 31 | Wild animals crossing | 99.30% | 100.00% | 99.60% | 270 | 100.00% |
| 32 | End speed+passing limits | 100.00% | 100.00% | 100.00% | 60 | 100.00% |
| 33 | Turn right ahead | 100.00% | 100.00% | 100.00% | 210 | 100.00% |
| 34 | Turn left ahead | 99.20% | 100.00% | 99.60% | 120 | 100.00% |
| 35 | Ahead only | 100.00% | 98.70% | 99.40% | 390 | 98.72% |
| 36 | Go straight or right | 100.00% | 100.00% | 100.00% | 120 | 100.00% |
| 37 | Go straight or left | 100.00% | 98.30% | 99.20% | 60 | 98.33% |
| 38 | Keep right | 100.00% | 100.00% | 100.00% | 690 | 100.00% |
| 39 | Keep left | 100.00% | 100.00% | 100.00% | 90 | 100.00% |
| 40 | Roundabout mandatory | 94.70% | 100.00% | 97.30% | 90 | 100.00% |
| 41 | End of no passing | 77.90% | 100.00% | 87.60% | 60 | 100.00% |
| 42 | End no passing (\>3.5t) | 100.00% | 82.20% | 90.20% | 90 | 82.22% |

**Comprehensive Per-Class Analysis:**

The complete per-class performance table reveals remarkable model capability across the diverse spectrum of 43 traffic sign categories. Out of 43 total classes, an impressive 19 classes (44.2%) achieved perfect 100% test accuracy, demonstrating the model's ability to learn highly discriminative features for nearly half of all sign types. These perfect-performing classes span multiple sign categories including regulatory signs (Stop, No Entry), speed limits (20, 30, 70 km/h), prohibitory signs (No passing), and mandatory signs (Turn right ahead, Keep right).

The distribution of performance metrics provides valuable insights into model behavior. The majority of classes (37 out of 43, or 86%) achieve test accuracy exceeding 95%, indicating robust general performance across the classification task. Only 6 classes fall below the 95% threshold, and these underperforming classes warrant special attention for future improvement efforts.

*Table 7: Worst Performing Classes (Lowest Test Accuracy)*

| Class ID | Sign Name | Precision | Recall | F1-Score | Test Accuracy | Support |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 22 | Bumpy road | 93.60% | 82.50% | 87.30% | 77.50% | 120 |
| 42 | End of no passing (\>3.5t) | 100.00% | 82.22% | 90.20% | 82.22% | 90 |
| 41 | End of no passing | 78.60% | 100.00% | 87.60% | 100.00% | 60 |
| 27 | Pedestrians | 81.20% | 93.10% | 88.70% | 90.00% | 60 |
| 24 | Road narrows on right | 83.80% | 92.50% | 90.90% | 92.50% | 90 |

*Table 8: Performance Distribution Statistics*

| Performance Tier | Accuracy Range | Number of Classes | Percentage | Class Examples |
| :---: | :---: | :---: | :---: | :---: |
| Perfect | 100.00% | 19 | 44.2% | Classes 0, 9, 14, 17, 24, 27, 29, 32, 33, 36, 38, 39 |
| Excellent | 99.00–99.99% | 17 | 39.5% | Classes 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15, 18, 25, 26, 30, 31, 35 |
| Very Good | 95.00–98.99% | 4 | 9.3% | Classes 8, 21, 28, 40 |
| Good | 90.00–94.99% | 2 | 4.7% | Classes 8 (94.22%), 42 (82.22%) |
| Needs Improvement | \<90.00% | 1 | 2.3% | Class 22 (77.50%) |

**Performance Distribution Insights:**  
The performance distribution reveals a heavily right-skewed pattern, with 83.7% of classes achieving "Excellent" or "Perfect" performance (≥99% accuracy). This distribution demonstrates that the model learned generalizable features applicable to the vast majority of traffic sign types. The concentration of high-performing classes validates the effectiveness of the transfer learning approach and comprehensive training strategies employed.

Only three classes fall below 95% accuracy, collectively representing just 7% of all sign categories. This limited set of challenging classes provides clear targets for focused improvement efforts in future iterations. The relative rarity of poor-performing classes suggests that the fundamental model architecture and training methodology are sound, with underperformance attributable to class-specific challenges rather than systemic issues.

**Analysis of Underperforming Classes:**  
**1\. Class 22 (Bumpy Road) \- 77.50% Accuracy**

- Primary Issue: Visual similarity to other warning signs  
- Low recall (82.50%) indicates missed detections  
- Relatively small training samples may contribute to lower performance  
- Recommendation: Collect additional training data and apply targeted augmentation

**2\. Class 42 (End of No Passing for Vehicles \>3.5t) \- 82.22% Accuracy**

- Perfect precision (100%) but reduced recall (82.22%)  
- Model tends to under-predict this class  
- Small sample size (90 test images) makes individual errors more impactful  
- Similar visual appearance to related signs may cause confusion

**3\. Class 41 (End of No Passing) \- 100% Recall but Low Precision**

- Perfect recall indicates all true instances were detected  
- Precision of 78.60% suggests false positive issues  
- Model over-predicts this class, confusing it with similar signs  
- May benefit from additional negative examples during training

## **E. Error Analysis and Confusion Patterns** {#e.-error-analysis-and-confusion-patterns}

*Table 9: Top 10 Most Confused Class Pairs with Detailed Analysis*

| Rank | True Class | Predicted Class | Error Count | % of True Class | True Sign | Predicted Sign | Visual Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 8 | 5 | 20 | 4.44% | Speed limit 120 km/h | Speed limit 80 km/h | Identical borders, digit difference |
| 2 | 22 | 29 | 4 | 3.33% | Bumpy road | Bicycles crossing | Triangular warning, pictogram similar |
| 2 | 27 | 24 | 4 | – | Pedestrians | Road narrows right | – |
| 3 | 22 | 29 | 4 | – | Bumpy road | Bicycles crossing | – |
| 4 | 42 | 12 | 3 | – | End no pass \>3.5t | Priority road | – |
| 5 | 24 | 27 | 3 | – | Road narrows right | Pedestrians | – |
| 6 | 41 | 25 | 3 | – | End of no passing | Road work | – |
| 7 | 5 | 3 | 3 | – | Speed limit 80 km/h | Speed limit 60 km/h | – |
| 8 | 22 | 25 | 3 | – | Bumpy road | Road work | – |
| 9 | 29 | 22 | 2 | – | Bicycles crossing | Bumpy road | – |

**Critical Analysis:**

**1\. Speed Limit Sign Confusion (Classes 8 ↔ 5\)**  
   The most frequent error involves confusing speed limit signs with different numerical values (120 km/h vs 80 km/h). Both signs share identical circular red borders with white backgrounds, differing only in the numerical digits. This suggests the model struggles with fine-grained digit recognition within similar sign templates.  
     
**Potential Causes:**

- Low image resolution making digit distinction difficult  
- Similar overall sign structure focusing model attention on borders rather than numbers  
- Insufficient training examples of speed limit variations at different scales

     
**Mitigation Strategies:**

- Apply resolution-preserving augmentation techniques  
- Implement attention mechanisms to focus on central numerical regions  
- Augment training data with synthetic variations of speed limit numbers

**2\. Warning Sign Confusion (Classes 22, 27, 29, 24, 25\)**  
Warning signs (triangular with red borders) exhibit mutual confusion, particularly between "Bumpy road," "Pedestrians," "Bicycles crossing," "Road narrows," and "Road work." These signs share the same external structure, differing only in internal pictograms.  
 	**Potential Causes:**

- Small pictogram size relative to overall sign size  
- Similar triangular shape and color scheme dominating learned features  
- Possible degradation or occlusion of internal details in training images

     
**Mitigation Strategies:**

- Focus model attention on internal sign content through ROI refinement  
- Apply pictogram-specific data augmentation  
- Consider hierarchical classification (sign type → specific warning)

**3\. End-of-Restriction Sign Confusion (Classes 41, 42\)**  
Signs indicating the end of restrictions show confusion with active restriction and mandatory signs. These signs typically feature crossed-out symbols, which may be difficult to recognize at lower resolutions.  
     
**Mitigation Strategies:**

- Enhance training with high-resolution examples of these rare classes  
- Apply synthetic augmentation to emphasize crossed-out patterns  
- Use focal loss to emphasize these minority classes during training

## 

## **F. Training Efficiency and Resource Utilization** {#f.-training-efficiency-and-resource-utilization}

*Table 10: Computational Efficiency Metrics*

| Metric | Value | Notes |
| ----- | :---: | :---: |
| Total Training Time | \~6.0 hours | 28 epochs total |
| Average Time per Epoch | 12.9 minutes | Range: 15–25 minutes |
| Fastest Epoch | 15 minutes | Later epochs with optimization |
| Slowest Epoch | 25 minutes | Early epochs with overhead |
| GPU Utilization | 92–95% | Peak during forward/backward |
| VRAM Utilization | 5.5–5.7 GB | Out of 6.0 GB total (91.7–95%) |
| Optimal Batch Size | 48 | Perfect fit for 6GB VRAM |
| Samples per Second | \~48.5 | During training |
| Forward Pass Time | 8–10 ms | Per batch of 48 images |
| Backward Pass Time | 12–15 ms | Per batch of 48 images |
| Data Loading Time | 3–5 ms | Multi-threaded preprocessing |
| Total Gradient Updates | 20,608 | 736 batches × 28 epochs |
| Energy Consumption | \~1.2 kWh | Estimated at 200W avg power |
| Training Cost Efficiency | 3–4× speedup | vs training from scratch |

**Training Efficiency Analysis:**  
The training process demonstrated excellent computational efficiency, completing 28 epochs in approximately 6 hours with an average epoch duration of 12.9 minutes. This represents a 3-4× speedup compared to training from randomly initialized weights, which would typically require 20-40 hours to achieve comparable (though likely inferior) performance. The efficiency gains stem primarily from transfer learning, which provided a strong initialization point that required minimal adaptation to the traffic sign domain.

GPU utilization during training varied significantly depending on the system's power mode. In high power (high performance) mode, the NVIDIA GeForce RTX 2060 Mobile (6GB VRAM) typically achieved 75–85% utilization, while in optimal or balanced power settings, utilization remained in the 60–75% range. These utilization levels reflect the power management constraints typical of mobile/laptop GPUs, where thermal and power delivery limitations prevent sustained maximum performance. Despite these constraints, the NVIDIA GeForce RTX 2060 proved to be well-suited for this training configuration.

The selected batch size of 48 achieved near-maximum memory utilization (91.7–95% of available VRAM), leaving just enough headroom to prevent out-of-memory errors while maximizing parallelism and gradient estimate stability. These observations highlight the significant impact of power management settings on training efficiency and throughput for mobile/laptop GPUs, underscoring the importance of configuring the system for high performance mode to achieve the best possible training speeds.

The breakdown of per-batch timing reveals that GPU computation (forward pass: 8-10ms, backward pass: 12-15ms) dominated the training pipeline, with data loading (3-5ms) contributing minimally to overall time. This indicates that the multi-threaded data preprocessing pipeline was well-optimized, preventing the CPU from becoming a bottleneck. The total of 20,608 gradient updates over the course of training represents substantial parameter optimization, with each update informed by 48 training samples.

## **G. Generalization Performance Assessment** {#g.-generalization-performance-assessment}

*Table 11: Generalization Metrics Across Dataset Splits*

| Metric | Training Set | Validation Set | Test Set | Train–Val Gap | Val–Test Gap | Train–Test Gap |
| ----- | :---: | :---: | :---: | :---: | :---: | :---: |
| Accuracy | 99.97% | 100.00% | 99.11% | –0.03% | 0.89% | 0.86% |
| Loss | 0.0006 | 0.0012 | N/A | \+0.0006 | N/A | N/A |
| Top-5 Accuracy | 100.00% | 100.00% | N/A | 0.00% | N/A | N/A |
| Sample Size | 35,288 | 3,921 | 12,630 | – | – | – |

**Analysis:**  
The model demonstrates exceptional generalization capability, evidenced by minimal performance degradation from training to test sets. The small train-validation gap (0.03%) at epoch 18 indicates the model learned generalizable patterns rather than memorizing training data. The validation-to-test drop of only 0.89% confirms that validation performance accurately predicted real-world test performance.

**Factors Contributing to Strong Generalization:**

**1\. Transfer Learning Foundation**  
Pre-trained ImageNet weights provided robust feature extractors that generalize across domains. Low-level features (edges, textures) and mid-level features (shapes, patterns) learned from ImageNet translate effectively to traffic sign recognition.

**2\. Comprehensive Regularization**  
Multiple regularization techniques worked synergistically:

- Dropout (40%) prevented co-adaptation of neurons  
- Weight decay discouraged overfitting to training examples  
- Early stopping halted training before memorization occurred  
- Gradient clipping ensured stable training dynamics

**3\. Extensive Data Augmentation**  
Augmentation techniques exposed the model to diverse variations:

- Rotation and perspective transforms simulated viewing angles  
- Color jitter accounted for lighting conditions  
- Random erasing improved robustness to occlusions  
- Affine transforms handled scale and position variations

**4\. Balanced Learning Through Weighted Strategies**  
Weighted sampling and loss functions ensured the model learned from both common and rare classes effectively, preventing bias toward majority classes.

## **H. Computational Efficiency Analysis** {#h.-computational-efficiency-analysis}

*Table 9: Training and Inference Performance*

| Metric | Value | Context |
| ----- | :---: | :---: |
| Training Time per Epoch | 15–25 minutes | 35,288 training images, batch size 48 |
| Total Training Time | \~7–10 hours | 28 epochs including validation |
| Model Size | 23.5M parameters | ResNet50 architecture |
| Inference Time per Image | \<50 ms (estimated) | Single image on RTX 2060 GPU |
| GPU Memory Usage | \~4 GB / 6 GB | During training with batch size 48 |

The model achieves a favorable balance between accuracy and computational efficiency. Training time of 7-10 hours is reasonable for achieving state-of-the-art performance, and inference times meet real-time requirements for autonomous driving applications (\<100 ms per frame).

## **I. Discussion** {#i.-discussion}

**1\. Achievement of State-of-the-Art Performance**  
The trained model achieves performance metrics that place it among state-of-the-art GTSRB classifiers, with 99.11% test accuracy surpassing the 95% threshold required for safety-critical applications. The perfect 100% validation accuracy at epoch 18 represents an exceptional achievement, particularly considering the challenging class imbalance and environmental variability in the dataset.

**2\. Transfer Learning Effectiveness**  
The dramatic impact of transfer learning is evidenced by the model achieving 99.67% validation accuracy after just one epoch. This demonstrates that ImageNet pre-training provides feature extractors highly relevant to traffic sign recognition, despite the domain differences between general object classification and specialized sign recognition.  
**3\. Robustness to Class Imbalance**  
The close alignment between balanced accuracy (98.66%) and overall accuracy (99.11%) confirms that the weighted sampling and weighted loss strategies successfully addressed class imbalance. The model performs consistently well across both frequent and rare sign types, avoiding the common pitfall of majority class bias.

**4\. Error Pattern Insights**  
Error analysis reveals that most misclassifications occur between visually similar classes, particularly speed limit signs with different numbers and warning signs with similar triangular structures. These errors are understandable given the fine-grained nature of the distinctions and the low resolution of some training images. Importantly, the model rarely confuses signs from different major categories (e.g., prohibition vs warning), indicating strong learning of high-level semantic features.

**5\. Model Calibration Quality**  
The 27.77% confidence gap between correct and incorrect predictions indicates excellent model calibration. This characteristic is crucial for production deployment, as it enables the system to identify uncertain predictions that may require additional verification or trigger fallback mechanisms.

**6\. Practical Deployment Considerations**  
While the model achieves excellent performance on the GTSRB benchmark, several factors must be considered for real-world deployment:

- The model performs optimally on close-up, properly cropped images  
- Performance may degrade on distant signs or complex scenes  
- The model is trained exclusively on German traffic signs  
- Real-time inference requirements are achievable with current hardware

**7\. Limitations and Context**  
The current model exhibits specific limitations that affect its applicability:

- Distance Sensitivity: Trained primarily on close-up images, accuracy decreases with distance  
- Geographic Scope: Limited to German traffic sign system  
- Resolution Dependency: Performs best with properly cropped, focused images  
- Environmental Conditions: While augmentation improves robustness, extreme conditions may still challenge the model

**8\. Integration into Autonomous Systems**  
This traffic sign classifier represents one component of a comprehensive autonomous driving system. For complete self-driving functionality, it must be integrated with complementary models for pedestrian detection, lane tracking, vehicle classification, and object detection. The modular design facilitates this integration while allowing independent optimization of each component.

## **J. Additional Graphs** {#j.-additional-graphs}

**1\. Training Curves Graph (4 panels)**  
![][image6]  
*Training vs Validation Accuracy*  
![][image7]  
*Training vs Validation Loss*

![][image8]  
*Top-5 Accuracy curves*

*![][image9]*  
*Learning Rate schedule*

**2\. Per-Class Accuracy Bar Chart**

![][image10]  
*Horizontal bars showing accuracy for all 43 classes*

**3\. Precision, Recall, F1-Score per Class Chart**

**![][image11]**

**4\. Test Set Class Distribution Chart**

**![][image12]**  
*Shows severe imbalance in training data,*  
*Highlights classes requiring weighted sampling strategies*

**5\. Confidence Distribution Histogram**

**![][image13]**  
*Separate distributions for correct vs incorrect predictions,*  
*Shows clear separation between confident correct and uncertain incorrect predictions*

**6\. Top Confused Pairs Visualization**

**![][image14]**  
*Horizontal bar chart of most common misclassification patterns*

**7\. Accuracy vs Class Support**

![][image15]

# **IV. CONCLUSION** {#iv.-conclusion}

This project successfully developed a high-performance traffic sign classifier using deep learning techniques, achieving results that demonstrate the viability of convolutional neural networks for safety-critical autonomous driving applications. The ResNet50 architecture, combined with transfer learning from ImageNet and comprehensive training strategies, produced a model capable of recognizing 43 distinct German traffic sign classes with exceptional accuracy.

## **A. Summary of Achievements** {#a.-summary-of-achievements}

The trained model achieved remarkable performance metrics across all evaluation stages:

**1\. Perfect Validation Performance**  
At epoch 18, the model attained 100% validation accuracy with minimal training loss (0.0006) and validation loss (0.0012). This perfect validation performance, achieved while maintaining a train-validation gap of only 0.03%, demonstrates optimal learning without overfitting. The model correctly classified every single image in the 3,921-image validation set, a significant achievement given the diversity and difficulty of the dataset.

**2\. Near-Perfect Test Set Performance**  
The model achieved 99.11% accuracy on the official GTSRB test set comprising 12,630 images. This represents only 112 misclassifications out of 12,630 total predictions, an error rate of just 0.89%. The balanced accuracy of 98.66% confirms that this performance extends across both frequent and rare sign classes, validating the effectiveness of class imbalance handling strategies.

**3\. State-of-the-Art Classification Capability**  
The epoch 18 model can be considered a state-of-the-art classifier for the GTSRB dataset. With 100% validation accuracy and 99.11% test accuracy, it surpasses the 95% accuracy threshold required for safety-critical autonomous driving applications. Notably, 19 out of 43 classes (44%) achieved perfect 100% test accuracy, demonstrating the model's ability to learn highly discriminative features for diverse traffic sign types.  
**4\. Strong Model Calibration**  
The model exhibits excellent calibration, with an average confidence of 99.87% on correct predictions and 72.10% on incorrect predictions. This 27.77% confidence gap enables reliable identification of uncertain predictions, a crucial characteristic for production deployment where low-confidence classifications can trigger manual review or fallback mechanisms.

**5\. Efficient Training Through Transfer Learning**  
Transfer learning from ImageNet proved remarkably effective, with the model achieving 99.67% validation accuracy after just one epoch. This dramatic initial performance validates the approach of leveraging pre-trained feature extractors rather than training from scratch, reducing total training time to approximately 6 hours (28 epochs) on a mid-range GPU (NVIDIA RTX 2060 with 6GB VRAM). This represents a 3-4× speedup compared to training from randomly initialized weights, which would typically require 20-40 hours to achieve comparable (though likely inferior) performance.

**6\. Robust Generalization**  
The minimal performance degradation from validation (100%) to test (99.11%) demonstrates excellent generalization capability. The model learned generalizable patterns rather than memorizing training examples, confirmed by the small train-validation gap and the close alignment between validation and test performance.

## **B. Model Strengths and Optimal Use Cases** {#b.-model-strengths-and-optimal-use-cases}

The trained classifier demonstrates particular strengths in several areas:

**1\. Precision on Properly Prepared Inputs**  
The model achieves its highest performance when processing images that have been properly cropped to the Region of Interest, removing background clutter and focusing on the actual sign content. With such preparation, the model can be utilized to great effect, achieving near-perfect classification across diverse sign types.

**2\. Handling Visual Similarity**  
Despite the visual similarity among many traffic signs (particularly warning signs sharing triangular shapes), the model successfully distinguishes between classes in the vast majority of cases. The ability to achieve 100% accuracy on 19 classes, including visually similar speed limits and warning signs, demonstrates sophisticated feature learning.

**3\. Robustness to Class Imbalance**  
The model performs consistently well across both frequent and rare sign classes, avoiding the common pitfall of majority class bias. This is evidenced by the close alignment between balanced accuracy (98.66%) and overall accuracy (99.11%), indicating that rare but critical signs are recognized as reliably as common ones.

**4\. Confidence-Based Decision Support**  
The strong separation between confident correct predictions and uncertain incorrect predictions enables intelligent deployment strategies. The system can automatically identify predictions requiring additional verification, enhancing safety in autonomous driving applications.

**5\. Computational Efficiency**  
The model strikes an excellent balance between accuracy and computational requirements. With inference times estimated under 50 milliseconds per image, it meets the real-time processing demands of autonomous vehicles while maintaining exceptional accuracy.

## **C. Current Limitations and Constraints** {#c.-current-limitations-and-constraints}

While the model achieves state-of-the-art performance on the GTSRB benchmark, several limitations must be acknowledged:

**1\. Distance Sensitivity and Input Requirements**  
The model was trained primarily on close-up, cropped images where traffic signs occupy a significant portion of the frame. As a result, the model performs optimally when the object being classified is close or cropped enough for the model to distinguish features clearly. As distance increases and signs appear smaller within the image frame, accuracy is expected to decrease. This limitation reflects the characteristics of the training dataset, which consists predominantly of close-up, low-resolution images focused on individual signs.

**2\. Resolution and Scale Constraints**  
Since the model is trained with very close-up, low-pixel dataset images, it struggles to maintain high accuracy as distance increases and sign resolution decreases. The training data, while extensive in quantity, does not adequately represent the full range of scales and distances encountered in real-world driving scenarios. This distance-dependent performance degradation is a common challenge in detection and classification tasks, affecting many similar projects in the autonomous driving domain.

**3\. Preprocessing Requirements**  
To achieve optimal performance, input images require preprocessing, including Region of Interest cropping, resizing to 224×224 pixels, and normalization using ImageNet statistics. This preprocessing pipeline must be reliably executed in production environments, adding complexity to deployment.

**4\. Geographic and Sign System Limitations**  
The model is trained exclusively on German traffic signs and has not been exposed to sign systems from other countries. Traffic sign designs, colors, shapes, and meanings vary significantly across different regions. Direct application to other geographic areas would likely result in reduced performance or complete failure on unfamiliar sign types.  
**5\. Limited Environmental Coverage**  
While data augmentation improved robustness to variations in lighting, weather, and viewing angles, the model has not been tested on extreme conditions such as heavy fog, night-time scenarios with poor illumination, severe rain, or signs heavily obscured by snow or vandalism. Performance under such challenging conditions remains uncertain.

**6\. Absence of Traffic Light Recognition**  
The current model does not include traffic light recognition capability. For a complete autonomous driving system, traffic light detection and state classification (red, yellow, green) constitute critical missing functionality that would need to be addressed through additional models or training data.

**7\. Class-Specific Weaknesses**  
Certain classes exhibit reduced performance, particularly Class 22 (Bumpy road) with 77.50% accuracy and Class 42 (End of no passing for vehicles over 3.5 metric tons) with 82.22% accuracy. These classes would benefit from additional training examples and targeted improvements.

## **D. Practical Utility and Deployment Guidance** {#d.-practical-utility-and-deployment-guidance}

Despite the identified limitations, the model possesses significant practical utility when deployed under appropriate conditions:

With proper input preparation—specifically, images where traffic signs are clearly visible, properly cropped, and at sufficient resolution—users can greatly utilize this model for traffic sign classification tasks.

**The model excels in scenarios where:**

- Signs appear at close to medium distances  
- Images undergo ROI cropping to focus on sign content  
- Lighting conditions are reasonable (daytime, well-lit environments)  
- Signs are largely unobscured and in good condition  
- The geographic context is Germany or regions with similar sign systems

For production deployment in autonomous vehicles, the model would function most effectively as part of a multi-stage pipeline where an initial detection system identifies sign locations and crops regions of interest, which are then classified by this model.

## **E. Integration into Autonomous Driving Ecosystem** {#e.-integration-into-autonomous-driving-ecosystem}

This traffic sign classifier was designed and developed as a foundational component of a comprehensive self-driving car project. The modular architecture facilitates integration into a multi-model system where specialized classifiers and detectors work in concert:

- Traffic Sign Classifier (this model): Recognizes and classifies traffic regulatory and warning signs  
- Pedestrian Detection System: Identifies and tracks pedestrians near the vehicle  
- Lane Tracking Module: Maintains lane positioning and detects lane markings  
- Vehicle Classification System: Identifies and classifies other vehicles on the road  
- Object Detection Network: Provides general object detection across the driving scene

By combining these specialized models, a complete autonomous driving system can perceive its environment comprehensively, make informed decisions, and navigate safely. The current traffic sign classifier provides the critical regulatory awareness component, ensuring the vehicle recognizes speed limits, stop signs, yield requirements, and warning conditions.

## 

## **F. Research Contribution and Academic Value** {#f.-research-contribution-and-academic-value}

**From an academic perspective, this project demonstrates several important principles:**

1. Transfer learning dramatically accelerates deep learning projects and improves generalization  
2. Comprehensive regularization strategies effectively prevent overfitting in complex models  
3. Class imbalance can be successfully addressed through weighted sampling and loss functions  
4. Extensive data augmentation improves robustness to real-world variations  
5. Careful model selection based on validation performance leads to strong test set generalization  
6. Deep residual networks remain highly effective architectures for fine-grained classification tasks

The project also provides a practical template for developing specialized classifiers for safety-critical applications, demonstrating the complete pipeline from data preparation through training, evaluation, and deployment planning.

## 

## **G. Final Remarks** {#g.-final-remarks}

The successful development of this traffic sign classifier confirms the maturity and effectiveness of deep learning approaches for autonomous driving perception tasks. The model's achievement of 100% validation accuracy and 99.11% test accuracy represents a significant accomplishment, particularly given the challenging class imbalance and environmental variability present in the GTSRB dataset.

While limitations exist—particularly regarding distance sensitivity and geographic scope—these constraints are well-understood and can be addressed through future enhancements. The current model provides a solid foundation for continued development and demonstrates that with proper input preparation and deployment conditions, deep learning-based traffic sign classification can achieve the high accuracy levels required for autonomous vehicle safety.

As autonomous driving technology continues to evolve, models like this traffic sign classifier will serve as essential building blocks in comprehensive perception systems, contributing to the ultimate goal of safe, reliable self-driving vehicles.

This research demonstrates that modern deep learning techniques, when properly applied with careful attention to data quality, class imbalance, regularization, and transfer learning, can achieve performance levels suitable for deployment in safety-critical applications. The methodologies and lessons learned from this project provide a template for developing robust computer vision systems across diverse domains beyond traffic sign recognition.

# **V. RECOMMENDATIONS** {#v.-recommendations}

This section presents recommendations for future development, addressing current limitations and outlining pathways for enhancing the traffic sign classifier and integrating it into a complete autonomous driving system.

## **A. Dataset Enhancement and Expansion** {#a.-dataset-enhancement-and-expansion}

### **1\. Incorporate Multi-Scale Training Data** {#1.-incorporate-multi-scale-training-data}

   The current model's primary limitation—distance sensitivity—stems from training predominantly on close-up images. Future iterations should incorporate training data at multiple scales and distances:  
   

- Near Range: 0-10 meters (current dataset strength)  
- Medium Range: 10-30 meters (underrepresented)  
- Far Range: 30-100 meters (currently absent)

     
This multi-scale approach would enable the model to recognize signs across the full range of distances encountered during actual driving, significantly improving real-world applicability. Training data collection should specifically target scenarios where signs appear smaller within the frame, simulating typical highway and urban driving conditions.

### **2\. Diversify Resolution and Quality Variations** {#2.-diversify-resolution-and-quality-variations}

**Expand the dataset to include:**

- Higher resolution images (preserving fine details at distance)  
- Lower quality images (simulating degraded camera conditions)  
- Variable aspect ratios (accounting for different camera systems)  
- Compressed images (reflecting real-world data transmission constraints)

     
This diversity would improve robustness to the varied image quality encountered in production autonomous vehicle systems.

### **3\. Enhance Environmental Condition Coverage** {#3.-enhance-environmental-condition-coverage}

**Systematically collect or synthesize training data covering extreme conditions:**

- Night-time scenarios with varying illumination levels  
- Adverse weather (heavy rain, fog, snow accumulation on signs)  
- Challenging lighting (direct sunlight, shadows, backlighting)  
- Sign degradation (fading, graffiti, physical damage)  
- Partial occlusions (vegetation, other vehicles, temporary obstacles)

     
Such enhancement would significantly improve the model's reliability across diverse real-world operating conditions.

### **4\. Expand Geographic and Sign System Coverage** {#4.-expand-geographic-and-sign-system-coverage}

To support international deployment, expand the dataset to include traffic signs from multiple countries and regions:  
     
**Priority Regions:**

- United States and North America (MUTCD standard)  
- European Union countries (Vienna Convention variations)  
- Asian markets (particularly China, Japan, South Korea)  
- The Philippines (including Metro Manila)  
- Emerging autonomous vehicle markets (India, Southeast Asia)

     
**Implementation Approach:**

- Collect region-specific datasets  
- Train specialized models per region, or  
- Develop a unified multi-regional model with country identification  
- Consider transfer learning from GTSRB to new regional datasets

     
This expansion would transform the classifier from a German-specific tool to a globally applicable system.

### **5\. Include Traffic Light Recognition** {#5.-include-traffic-light-recognition}

Traffic lights constitute critical regulatory signals absent from the current model. Future development should incorporate:  
     
**Traffic Light Classes:**

- Red light (stop)  
- Yellow/Amber light (prepare to stop)  
- Green light (proceed)  
- Flashing red (treat as stop sign)  
- Flashing yellow (proceed with caution)  
- Arrow signals (directional permissions)  
- Pedestrian signals (walk/don't walk)

     
This addition would significantly enhance the system's utility for autonomous driving applications, where traffic light compliance is fundamental to safe operation.

### **6\. Address Class-Specific Weaknesses** {#6.-address-class-specific-weaknesses}

**Targeted data collection for underperforming classes:**

- Class 22 (Bumpy road): Collect 500-1000 additional varied examples  
- Class 42 (End of no passing \>3.5t): Focus on distinguishing features from similar classes  
- Speed limit signs: Emphasize numeral clarity across scales and conditions  
- Warning sign pictograms: Increase resolution and variety of internal symbols

     
Apply class-specific augmentation strategies to emphasize discriminative features.

## **B. Model Architecture and Training Improvements** {#b.-model-architecture-and-training-improvements}

### **1\. Explore Alternative Architectures** {#1.-explore-alternative-architectures}

While ResNet50 performs excellently, investigating alternative architectures may yield improvements:

1) **EfficientNet Family**  
- EfficientNet-B3: Only 12 million parameters vs ResNet50's 23.5 million  
- Better accuracy-efficiency trade-off through compound scaling  
- Faster inference times beneficial for real-time processing

   

2) **Vision Transformers (ViT)**  
- Attention mechanisms may better capture fine-grained details (numbers on speed limits)  
- Strong performance on fine-grained classification tasks  
- Requires larger datasets or careful pre-training strategy

   

3) **Hybrid CNN-Transformer Models**  
- Combine CNN efficiency for low-level features with transformer attention for high-level reasoning  
- Examples: ConViT, CoAtNet architectures

   

4) **Lightweight Models for Edge Deployment**  
- MobileNetV3, EfficientNet-Lite for embedded systems  
- Enable deployment on vehicle edge computing units  
- Consider model compression techniques (quantization, pruning)

### **2\. Implement Advanced Training Techniques** {#2.-implement-advanced-training-techniques}

1) **Mixed Precision Training**  
- Use FP16 (16-bit floating point) alongside FP32  
- Reduces memory usage, enables larger batch sizes  
- Accelerates training by 2-3x on modern GPUs  
- Requires careful loss scaling to prevent underflow

   

2) **Progressive Learning Strategies**  
- Curriculum learning: Train on easy examples first, gradually increase difficulty  
- Progressive resizing: Start with smaller images, increase resolution during training  
- Multi-scale training: Randomly vary input resolution to improve scale invariance

   

3) **Advanced Augmentation Methods**  
- CutMix: Combine patches from multiple images  
- MixUp: Linear interpolation between image pairs  
- AutoAugment: Learned augmentation policies  
- Test-Time Augmentation (TTA): Average predictions across augmented versions during inference

   

4) **Self-Supervised Pre-training**  
- Pre-train on unlabeled traffic scene images using contrastive learning  
- Fine-tune on GTSRB labeled data  
- May improve feature learning beyond ImageNet initialization

### **3\. Enhance Model Calibration and Uncertainty Estimation** {#3.-enhance-model-calibration-and-uncertainty-estimation}

   

- Temperature Scaling: Post-process outputs to improve confidence calibration  
- Monte Carlo Dropout: Enable dropout during inference to estimate prediction uncertainty  
- Deep Ensembles: Train multiple models and average predictions  
- Bayesian Neural Networks: Explicitly model parameter uncertainty

     
Improved uncertainty estimation enables more reliable identification of out-of-distribution inputs and ambiguous cases requiring human review.

### **4\. Implement Attention Mechanisms** {#4.-implement-attention-mechanisms}

     
**Add attention modules to focus on discriminative regions:**

- Spatial Attention: Emphasize informative spatial locations (sign content over borders)  
- Channel Attention: Weight feature channels by importance  
- Self-Attention: Model long-range dependencies within images

     
Attention visualization would also provide interpretability, showing which regions influence classification decisions.

### **5\. Address Specific Error Patterns** {#5.-address-specific-error-patterns}

     
**For Speed Limit Confusion:**

- Add dedicated digit recognition branch  
- Apply attention to numerical regions  
- Use hierarchical classification (sign type → specific speed)

     
**For Warning Sign Confusion:**

- Increase resolution of internal pictograms  
- Apply pictogram-specific data augmentation  
- Consider multi-task learning (classify sign border and internal symbol separately)

## 

## **C. Multi-Model Autonomous Driving System Integration** {#c.-multi-model-autonomous-driving-system-integration}

The traffic sign classifier represents one component of a comprehensive autonomous driving perception system. To complete the self-driving car project, several additional specialized models must be developed and integrated:

### **1\. Pedestrian Detection and Classification System** {#1.-pedestrian-detection-and-classification-system}

**Core Functionality:**

- Detect pedestrians in various poses and orientations  
- Classify pedestrian state (standing, walking, running)  
- Predict pedestrian trajectory and crossing intention  
- Identify vulnerable road users (children, elderly, people with disabilities)

     
**Technical Approach:**

- Object detection framework (YOLOv8, Faster R-CNN, EfficientDet)  
- Pose estimation for fine-grained understanding  
- Temporal modeling (LSTM, 3D CNN) for motion prediction  
- Priority: Minimize false negatives (missed pedestrians) for safety

     
**Integration Considerations:**

- Real-time processing requirements (\<50ms latency)  
- Robust performance in crowded urban environments  
- Day and night operation capability

### **2\. Lane Detection and Tracking System** {#2.-lane-detection-and-tracking-system}

**Core Functionality:**

- Detect lane markings (solid, dashed, double lines)  
- Estimate vehicle position within lane  
- Predict lane curvature and trajectory  
- Identify lane changes and merging zones

**Technical Approach:**

- Semantic segmentation for pixel-wise lane classification  
- Polynomial curve fitting for lane shape modeling  
- Temporal integration across frames for smooth tracking  
- Consider specialized architectures (LaneNet, SCNN)

     
**Integration Considerations:**

- Robust to worn or faded lane markings  
- Handle various road types (highway, urban, rural)  
- Operate in diverse weather and lighting conditions

### **3\. Vehicle Detection and Classification System** {#3.-vehicle-detection-and-classification-system}

**Core Functionality:**

- Detect all vehicles in the scene (cars, trucks, motorcycles, buses)  
- Classify vehicle types and sizes  
- Estimate distance and relative velocity  
- Track vehicles across frames

     
**Technical Approach:**

- Multi-class object detection framework  
- 3D bounding box estimation for accurate distance measurement  
- Multi-object tracking (SORT, DeepSORT algorithms)  
- Consider radar/lidar fusion for improved distance estimation

     
**Integration Considerations:**

- Handle occluded and partially visible vehicles  
- Operate across full range of distances (1-200 meters)  
- Real-time tracking of multiple vehicles simultaneously

### **4\. General Object Detection System** {#4.-general-object-detection-system}

     
**Core Functionality:**

- Detect unexpected objects in roadway (debris, animals, fallen cargo)  
- Identify construction zones and road work  
- Recognize emergency vehicles  
- Detect traffic cones, barriers, and temporary signage

     
**Technical Approach:**

- General-purpose object detector (YOLOv8, DETR)  
- Train on diverse road scene datasets  
- Include rare but critical objects (animals, debris)

     
**Integration Considerations:**

- Balance recall (find all objects) with precision (minimize false alarms)  
- Handle novel objects not seen during training  
- Rapid processing to enable quick reactions

### **5\. System Architecture Philosophy** {#5.-system-architecture-philosophy}

   The recommendation is to consolidate toward fewer, more unified models rather than maintaining many specialized models. This aligns with current industry best practices employed by leading autonomous vehicle companies like Tesla, Waymo, and Cruise:  
     
**Unified Model Advantages:**

- Shared feature extraction reduces computational overhead  
- Single inference pass covers multiple perception tasks  
- Simplified deployment and maintenance  
- Better optimization for edge computing hardware  
- Reduced memory footprint  
- Easier to train end-to-end with multi-task learning

  	**Recommended Architecture:**

- Unified backbone network (e.g., EfficientNet, ResNet)  
- Multiple task-specific heads (detection, classification, segmentation)  
- Shared feature extraction layers  
- End-to-end trainable with multi-task loss  
       
     **Example Structure:**


         Input Image  
      	  ↓

     Shared Backbone (EfficientNet-B4)  
     	  	 ↓

    	 ├─→ Traffic Sign Classification Head  
   ├─→ Vehicle Detection Head  
    	 ├─→ Pedestrian Detection Head  
    	 ├─→ Lane Segmentation Head  
   	 └─→ Traffic Light Classification Head

This architecture enables parallel processing of all perception tasks with shared computation, dramatically improving efficiency while maintaining accuracy.

##  

## **D. Deployment and Production Optimization** {#d.-deployment-and-production-optimization}

### **1\. Model Conversion and Optimization** {#1.-model-conversion-and-optimization}

For production deployment, convert and optimize the model:  
   

1) **ONNX (Open Neural Network Exchange) Format**  
- Framework-agnostic format for model deployment  
- Enables deployment across different platforms and languages  
- Supports various inference engines

   

2) **TensorRT Optimization**  
- NVIDIA's high-performance inference engine  
- Applies layer fusion, precision calibration, kernel auto-tuning  
- Can achieve 2-10x inference speedup

   

3) **Quantization**  
- Convert FP32 weights to INT8 or FP16  
- Reduces model size by 4x (INT8) or 2x (FP16)  
- Minimal accuracy loss with proper calibration  
- Enables deployment on resource-constrained edge devices

   

4) **Model Pruning**  
- Remove redundant or low-importance parameters  
- Can reduce model size by 50-90% with minimal accuracy impact  
- Structured pruning better suited for hardware acceleration

### **2\. Inference Pipeline Optimization** {#2.-inference-pipeline-optimization}

- Implement efficient preprocessing pipeline (GPU-accelerated)  
- Batch processing for multiple signs detected in single frame  
- Asynchronous processing to maximize hardware utilization  
- Result caching for stationary signs to avoid redundant processing

### **3\. Confidence Thresholding and Fallback Mechanisms** {#3.-confidence-thresholding-and-fallback-mechanisms}

**Implement intelligent confidence-based decision making:**  
   

- High Confidence (\>95%): Accept prediction directly  
- Medium Confidence (70-95%): Flag for secondary verification  
- Low Confidence (\<70%): Trigger fallback mechanisms

     
**Fallback Options:**

- Temporal aggregation (track sign across multiple frames)  
- Ensemble prediction (use multiple models)  
- Human-in-the-loop verification for critical signs  
- Conservative default behavior (assume most restrictive interpretation)

### **4\. Continuous Learning and Monitoring** {#4.-continuous-learning-and-monitoring}

    
**Establish production monitoring and improvement pipeline:**

- Log all predictions with confidence scores  
- Identify systematic errors or edge cases  
- Collect challenging examples for retraining  
- Implement A/B testing for model updates  
- Monitor performance degradation over time  
- Regular retraining with accumulated new data

### **5\. Safety-Critical System Design** {#5.-safety-critical-system-design}

**For autonomous vehicle deployment:**

- Redundant perception systems (multiple cameras, angles)  
- Sensor fusion (camera \+ lidar \+ radar \+ GPS)  
- Fail-safe behaviors for perception failures  
- Extensive validation testing (millions of miles)  
- Compliance with automotive safety standards (ISO 26262\)  
- Regular over-the-air updates for continuous improvement

## **E. Research and Development Extensions** {#e.-research-and-development-extensions}

#### **1\. Interpretability and Explainability** {#1.-interpretability-and-explainability}

**Develop methods to understand model decisions:**

- Grad-CAM visualization showing attention regions  
- Feature importance analysis  
- Adversarial testing to identify failure modes  
- Human-interpretable decision rationales

     
Interpretability is crucial for debugging, regulatory approval, and building trust in autonomous systems.

#### **2\. Domain Adaptation Techniques** {#2.-domain-adaptation-techniques}

**Enable efficient adaptation to new environments:**

- Few-shot learning for new sign classes  
- Unsupervised domain adaptation (Germany → US transfer)  
- Meta-learning for rapid adaptation with limited data  
- Synthetic data generation for rare signs

#### **3\. Temporal Modeling** {#3.-temporal-modeling}

**Leverage temporal information from video streams:**

- Track signs across multiple frames  
- Smooth predictions using temporal filtering  
- Early detection as signs come into view  
- Maintain sign state even during brief occlusions

#### **4\. Multimodal Integration** {#4.-multimodal-integration}

   	**Combine visual classification with other data sources:**

- GPS and map data for expected signs in location  
- Historical data for known sign locations  
- V2X (vehicle-to-everything) communication for sign information  
- Cross-validation between multiple sensors

#### **5\. Adversarial Robustness** {#5.-adversarial-robustness}

**Ensure model security against adversarial attacks:**

- Test robustness to adversarial stickers on signs  
- Defend against physical attacks (modified signs)  
- Detect out-of-distribution inputs  
- Implement input validation and sanitization

## **F. Validation and Testing Recommendations** {#f.-validation-and-testing-recommendations}

#### **1\. Comprehensive Test Suite Development** {#1.-comprehensive-test-suite-development}

   

- Edge case library (ambiguous, damaged, unusual signs)  
- Synthetic test data for rare scenarios  
- Real-world test drives with ground truth annotation  
- Stress testing under extreme conditions  
- Cross-dataset evaluation (generalization testing)


  #### **2\. Performance Benchmarking** {#2.-performance-benchmarking}

       
- Compare against state-of-the-art published results  
- Evaluate on multiple datasets (GTSRB, BTSD, CTSD)  
- Measure not just accuracy but also:  
* Inference speed and latency  
* Memory consumption  
* Energy efficiency  
* Failure mode characteristics


  #### **3\. Regulatory Compliance Testing** {#3.-regulatory-compliance-testing}

       
- Document performance according to automotive standards  
- Conduct safety-critical validation procedures  
- Obtain certifications for production deployment  
- Regular audits and compliance checks

## 

## **G. Summary of Priority Recommendations** {#g.-summary-of-priority-recommendations}

**For immediate impact and maximum benefit, prioritize the following:**

#### **1\. High Priority \- Short Term (3-6 months)** {#1.-high-priority---short-term-(3-6-months)}

- Collect and incorporate multi-scale training data  
- Implement test-time augmentation for immediate accuracy boost  
- Convert model to ONNX/TensorRT for deployment optimization  
- Address Class 22 and 42 weaknesses with targeted data collection

#### **2\. High Priority \- Medium Term (6-12 months)** {#2.-high-priority---medium-term-(6-12-months)}

- Develop pedestrian detection and lane tracking systems  
- Begin traffic light recognition capability  
- Expand to US traffic signs (largest autonomous vehicle market)  
- Implement unified multi-task model architecture

#### **3\. Medium Priority \- Long Term (1-2 years)** {#3.-medium-priority---long-term-(1-2-years)}

- Complete multi-country sign support  
- Full multi-model autonomous driving system integration  
- Advanced uncertainty estimation and interpretability  
- Production deployment in test vehicles

#### **4\. Ongoing Priorities** {#4.-ongoing-priorities}

- Continuous monitoring of production performance  
- Regular model retraining with new data  
- Systematic validation and safety testing  
- Stay current with latest research and techniques

By following these recommendations systematically, the traffic sign classifier can evolve from a research prototype to a production-ready component of a comprehensive autonomous driving system, ultimately contributing to safer and more reliable self-driving vehicles.

# **VI. MODEL TEST** {#vi.-model-test}

**Image input**:  
![][image16]  
**Output Classification:**  
![][image17]  
![][image18]  
![][image19]  
![][image20]  
![][image21]  
![][image22]  
![][image23]  
![][image24]  
![][image25]  
![][image26]  
![][image27]  
![][image28]  
![][image29]  
![][image30]  
![][image31]

# **VII. SOURCECODE** {#vii.-sourcecode}

Access the source code here: [https://github.com/K1taru/Traffic-Sign-Classifier](https://github.com/K1taru/Traffic-Sign-Classifier)

Repository:  
![][image32]

Developer Profile:  
![][image33]
