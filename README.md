# Gesture_Recognition_CNN

This is a project using CNN to classify three hand gestures in the Paper-Rock-Scissors game. 

Process and Repo File Walkthrough:

* Self_preparing_data.ipynb: notebook for generating datasets by consulting with your computer's camera, using cv2 module.

* removeBG.py: .py file for extracting hands by removing noisy background pixels (Caveats: if you take the photo with head behind your hand, removeBG may not recognize the head, leave the filtered picture with a round shape, which will most likely be recognized as a rock)

* VGG.ipynb: VGG-16 model implemented from scratch

* Revised_Resnet.ipynb: resnet-50 model implements from scratch

* sltDemo.py: .py file for the streamlit web-app demo of this project. See below the way to run demo

Run the following command to run demo:
streamlit run [path to the 'sltDemo.py' file]
