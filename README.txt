1. the libraries required to run the project including the full version of each library.

matplotlib                        3.8.0
numpy                             1.26.4
opencv-python                     4.10.0.84
scikit-image                      0.22.0
scikit-learn                      1.2.2
pillow                            10.2.0
torch                             2.5.1+cu118


2. how to run your code and where to look for the output files.

If there are models trained (There should be a folder named data/salveazaFisiere where I save the 2 models):
	a. Open Cava2 as a new project.
	b. In Parameters.py script line 10 there is the dir_test_examples variable for the path of the 200 test photos, change the path to the right one.
	c. Run the RunProject.py script. This script does both tasks, saving all the results in Solutie/331_Chitu_Tudor/Task1 and Solutie/331_Chitu_Tudor/Task2 folders

If there are no models trained (it is not the case, the steps below are for training, if no models loaded):
	a. Make sure that there is a folder named salveaza, where there are 3 folders, one named characters, one named fete and one named negative. In each of those there are pictures for 	negative and positive examples. In characters folder there are 4 folders, one for each character. If these folders are empty, the RunProject.py script will generate these pictures. 
	b. Make sure that there are the 2 models used saved in the folder salveazaFisirere with the names: best_cnn_model.pth and best_cnn_model_characters.pth . If not, run the 	RunRecognition.py script to train the recognition model. The detection model is trained by running the RunProject.py script if not loaded.
	c. Run the RunProject.py script. This script does both tasks, saving all the results in Solutie/331_Chitu_Tudor/Task1 and Solutie/331_Chitu_Tudor/Task2 folders


PS: The code is not clean at all. It should work only only from RunProject.py and FacialDetector.py if I simplify some functions, so there are a couple of junk files that could be deleted by adding a line of code, but if it works, it works. Didn't want to risk breaking it.
