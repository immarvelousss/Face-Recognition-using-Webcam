## Project Overview
This project recognizes an unknown face that is not within the dataset and notifies via email with a captured image and video of the unknown person

## Steps for this project:

1. Headshots.py 
	- create a directory under the dataset directory and name it as your name
	- in the headshots.py, replace the name variable with the newly created directory's name
	- run the program
	- click on the keyboard's space bar to take as many pictures as needed (the recommended number is about 20 pictures)
	- click on the escape key to exit the program

2. Run train_model.py
	- this would take the pictures in the dataset directory and train it

3. facial_req_email_on unknown faces with video
	- replace the email address of the email_sender and email_receiver variables
	- replace the email_sender's app password
	- run the program

## Attribution
This project is based on or uses code originally created by [Caroline Dunn](https://github.com/carolinedunn/facial_recognition). The original code is licensed under the MIT License.
