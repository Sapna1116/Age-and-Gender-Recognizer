### Objective
To build a gender and age detector that can approximate the gender and age of a person (face) in an image or through a webcam using Deep Learning on the Adience dataset.

### About the Project
This Python-based project utilizes Deep Learning to accurately identify the gender and age of a person from a single image of their face. The predicted gender can be either ‘Male’ or ‘Female’, and the predicted age is classified into one of the following ranges:
- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100)
The models are trained by Tal Hassner and Gil Levi, and this project addresses the challenges of accurately guessing age due to factors like makeup, lighting, obstructions, and facial expressions, making it a classification task rather than regression.

### Dataset
The project uses the Adience dataset, which includes:
- 26,580 images of 2,284 subjects
- 8 distinct age ranges
- Real-world imaging conditions such as noise, lighting, and pose

### Additional Python Libraries Required
OpenCV: Install using pip install opencv-python

### Project Files
- opencv_face_detector.pbtxt: Face detection model in text format.
- opencv_face_detector_uint8.pb: Face detection model in binary format.
- age_deploy.prototxt: Configuration file for the age detection CNN.
- age_net.caffemodel: Trained model for age detection.
- gender_deploy.prototxt: Configuration file for the gender detection CNN.
- gender_net.caffemodel: Trained model for gender detection.
- detect.py: Main script for detection.

### Steps to Follow
- Face Detection with Haar Cascade: Use OpenCV’s Haar cascade to detect faces in the image.

- Gender Recognition with CNN: Implemented using OpenCV’s DNN module and models trained by Gil Levi and Tal Hassner.

- Age Recognition with CNN: Utilize a CNN model to classify age into 8 predefined ranges. The model uses two files:
  -> deploy_agenet.prototxt (network configuration)
  -> age_net.caffemodel (trained model).

### Usage
-> Clone the Repository
-> Detect Gender and Age from an Image: python detect.py --image <image_name>
-> Detect Gender and Age using Webcam: python detect.py
Press Ctrl + C to stop the program.

### Examples
  python detect.py --image girl1.jpg
  Gender: Female | Age: 25-32 years
  
  python detect.py --image boy1.jpg
  Gender: Male | Age: 4-6 years
  
  python detect.py --image girl2.jpg
  Gender: Female | Age: 38-43 years
  
  python detect.py --image boy2.jpg
  Gender: Male | Age: 25-32 years
  
  python detect.py --image group1.jpg
  Gender: Female, Male, Male | Age: 25-32, 0-2, 8-12 years

### Conclusion
The project demonstrates the application of deep learning techniques to classify gender and age based on face images, using state-of-the-art CNN models and real-world datasets for robust performance.
