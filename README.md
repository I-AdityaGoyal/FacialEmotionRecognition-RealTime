# Facial Recognition and Emotion Analysis in Real-Time

This is a real-time facial recognition and emotion analysis project using transfer learning with VGG16 for feature extraction and deep face for emotional analysis. It utilizes OpenCV to process video feeds and analyze emotions in real-time.

## Folder Structure

```plaintext
|-- haarcascade_frontalface_default.xml
|-- FacialEmotionRecognition.py
|-- model.h5 # Make sure you add your model in the Directory!
|-- README.md
|-- requirements.txt
```

1. Clone this repository:
```bash
https://github.com/I-AdityaGoyal/FacialEmotionRecognition-RealTime.git
cd FacialRecognition-EmotionAnalysis
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Place your trained model.h5 file in the same directory.

4. Run the FacialEmotionRecognition.py script for real-time emotion analysis:
```bash
python FacialEmotionRecognition.py
```

