Deepfake Detection via ResNet50
a project by Md Tariqul Islam

Note on the Dataset: Due to GitHub's file size limitations, the 120,000+ training images (Celeb-DF and 140k Real/Fake Faces) are not included in this repository.
Links to the datasets can be found here:


celeb DF:
https://github.com/yuezunli/celeb-deepfakeforensics


140k real and fake faces (kaggle):
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces



How to test the model:
You do not need the dataset to run the application. The fully trained ResNet50 weights are saved in best_model.pth. 

Simply open app.py, enter streamlit run in the terminal and upload a test image to the web interface that opens up,
you can also use the test images uploaded with this project.

How to retrain the model:
If you wish to run training.py from scratch, please download the Celeb-DF dataset and place the extracted frames into a local dataset/ directory before running the script.
