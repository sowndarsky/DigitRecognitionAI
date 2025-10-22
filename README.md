🧠 Handwritten Digit Recognition (MNIST using CNN)
📘 Project Overview

This project builds a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset.
It demonstrates the power of deep learning in image classification tasks and serves as an introduction to Computer Vision and Neural Networks.

The model achieves ~99% accuracy, making it an excellent example of how AI can learn to understand and process visual data.

🎯 Objective

To develop an AI model capable of identifying handwritten digits from images using deep learning techniques.

🧩 Approach

Dataset: MNIST (70,000 images of digits from 0–9).

Preprocessing: Normalization and reshaping of images.

Model: Convolutional Neural Network (CNN) built using TensorFlow / Keras.

Training: The model is trained over multiple epochs to learn digit patterns.

Evaluation: Tested on unseen data to check accuracy and loss.

Visualization: Accuracy and loss graphs plotted for better understanding.

⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Jupyter / VS Code

🧠 Skills Learned

Image preprocessing and normalization

CNN architecture and training

Model evaluation and visualization

Understanding overfitting and model tuning

📊 Results
Metric	Value
Training Accuracy	~99.2%
Validation Accuracy	~99.0%
Test Accuracy	~99.06%

Model Accuracy and Loss Graphs:
(Include your screenshot of the graphs here — e.g., model_accuracy_loss.png)

🌍 Real-World Applications

Banking: Reading handwritten cheques and account numbers (OCR)

Postal Services: ZIP code and address recognition

Education: Digitizing handwritten answer sheets

Mobile Apps: Scanning and recognizing handwritten notes or numbers

🧩 Example Output
Epoch 5/5
1875/1875 [====] - 8s/step - accuracy: 0.9921 - loss: 0.0231 - val_accuracy: 0.9908 - val_loss: 0.0298

Test accuracy: 0.9906

🏁 Conclusion

This project demonstrates how Convolutional Neural Networks can effectively recognize handwritten digits with extremely high accuracy.
It serves as a foundation for building more advanced Optical Character Recognition (OCR) systems and real-world AI applications.
