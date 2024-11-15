# CryptoMancers: AI/ML-Driven Cryptographic Algorithm Identification  

## Project Overview  
CryptoMancers is an innovative AI/ML-based solution designed to identify cryptographic algorithms from ciphertext. By leveraging machine learning and deep learning models, the project aims to classify algorithms accurately, supporting diverse encryption methods including symmetric, asymmetric, and transposition techniques.  

This tool is targeted at security experts, organizations, and researchers aiming to enhance cryptographic security and reduce vulnerabilities.  

---

## Problem Statement  
Cryptographic algorithms play a vital role in securing data. Identifying the specific algorithm used, based only on the ciphertext, is a complex challenge. This project provides a cutting-edge solution to classify cryptographic techniques under various conditions with a high degree of accuracy.  

---

## Key Features  
- **Support for Multiple Cryptographic Techniques:**  
  - Symmetric encryption algorithms (e.g., AES, DES) with **80.47% accuracy**.  
  - Asymmetric encryption methods (e.g., RSA) with **71.36% accuracy**.  
  - Transposition techniques identified with **62% accuracy**.  
- **Custom Dataset:** Extensive dataset of ciphertexts generated for different algorithms.  
- **Advanced Models:** Gradient Boosting and Random Forest classifiers used for prediction.  
- **User-Friendly Interface:** Developed using **Streamlit**, providing an interactive platform for users to predict encryption algorithms efficiently.  

---

## Technologies Used  

### Backend  
- **Python & Flask**: Request handling and API integration.  
- **PyCryptodome**: Ciphertext generation.  

### Frontend  
- **Streamlit**: Interactive and user-friendly interface for algorithm prediction.  

### Machine Learning Models  
- **Scikit-learn**: Random Forest and Gradient Boosting classifiers.  
- **PyTorch**: Hash function modeling.  

---

## Model Development Workflow  
1. **Dataset Preparation**: Diverse ciphertext samples generated for various algorithms.  
2. **Model Training**: Implemented multiple ML models, optimizing for accuracy and performance.  
3. **Evaluation**: Models tested on different encryption methods achieving the following milestones:  
   - Symmetric encryption: **81.47% accuracy**.  
   - Asymmetric encryption: **71.36% accuracy**.  

---

## Results and Impact  

### Results  
- Outperformed baseline models like HKNNRF (73%) with 81.47% accuracy for symmetric algorithms.  
- Support for 14 encryption algorithms with comprehensive classification capabilities.  

### Impact  
- **Security Advancements**: Reduces risks of breaches by identifying weaknesses in cryptographic systems.  
- **Regulatory Compliance**: Ensures cryptographic security aligns with legal standards.  
- **Cost Efficiency**: Cost-effective solution using open-source tools and custom datasets.  

---

## Feasibility Analysis  

### Technical Viability  
- Leveraged robust ML libraries like Scikit-learn and PyTorch for efficient modeling.  
- Implemented open-source tools ensuring cost-effective solutions.  

### Challenges Overcome  
- **Diverse Dataset**: Addressed variability in encryption types by using a comprehensive custom dataset.  
- **Accuracy Variability**: Minimized inconsistencies in prediction accuracy with optimized models.  

---

## Future Scope and Vision  
1. **Hash Function Integration**: Plan to incorporate hash-based encryption algorithms for broader coverage.  
2. **Algorithm Strength Assessment**: Enable the identification of strengths and weaknesses of various encryption techniques.  
3. **Enhanced Security Innovation**: Advance cybersecurity practices to safeguard the digital world.  
4. **Real-Time Prediction**: Extend the solution for real-time encryption algorithm detection in active systems.  
5. **Scalability**: Expand model compatibility to support a wider range of cryptographic algorithms and datasets.  

---

## Research and References  
- [Cryptographic Algorithm Identification Using AI](https://pdfs.semanticscholar.org/a13c/948ea6bec848699decb0254ef5b7704d8a38.pdf)  
- [ML Techniques for Cryptographic Analysis](https://www.sciencedirect.com/science/article/pii/S0957417422006811)  
- [Introduction to Cryptographic Algorithm Identification](https://www.researchgate.net/publication/337940145)  

---

## How to Use  
1. Clone this repository to your local machine.  
2. Install dependencies from the `requirements.txt` file.  
3. Launch the **Streamlit** interface by running:  
   ```bash  
   streamlit run app.py  
   ```  
4. Input a ciphertext and receive predictions about the encryption algorithm used.  

---

## Contributing  
Contributions are welcome! Please submit a pull request or open an issue for any bugs or enhancements.  

---

## License  
This project is licensed under the MIT License.  

---  
## WARNING 
"It is not for public use that's the reason we don't habe any dedicated front-end panel."
