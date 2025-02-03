# GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration
Gaussian Splatting Avatar Creation Release 

This repo releases the codes for our avatar creating for a monocular video in ~ 40 mins. We also provide the Unity Editor for VR/AR application design using trained avatar.
## Install

## Data Gathering 
Please record the video by rotating the yourself slowly and keeping the camera stable. The time would be about 20 secs. Please make sure there is only one human inside a video each of the time.

## Data Preprocessing 
For capturing SMPLX parameters from video frames, please run our preprocssing using commands below:
###### conda activate data_preprocess
###### cd preprocess/
###### python run.py --root_path {ROOT_PATH} --gender {GENDER}
## Avatar Training 
###### conda activate avatar_training
###### cd  avatar_training/
###### python main.py --base=./configs/gaussians_docker_custom.yaml
## Results
The visual results will be available in Results/SubjectID. Please move state.json and avatar.ply to training_asserts/ in Unity Editor package.
