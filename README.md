# GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration
Gaussian Splatting Avatar Creation Release 

This repo releases our end-to-end pipeline for gaussian splatting avatar creation for a monocular video in ~ 40 mins. Our pipeline incorporates a novel Gaussian splatting technique with
customized preprocessing, enabling detailed facial expression reconstruction and seamless integra-
tion with Unity-based VR/AR platforms. Additionally, we present a Unity-integrated Gaussian
Splatting Avatar Editor, offering a user-friendly environment for VR/AR application develop-
ment. 
## Install

## Data Gathering 
Please record the video by rotating the yourself slowly and keeping the camera stable. The time would be about 20 secs. Please make sure there is only one human inside a video each of the time.

## Data Preprocessing 
For capturing SMPLX parameters from video frames, please run our preprocssing using commands below （Note: for infant mode, gender is INFANT）:
###### conda activate data_preprocess
###### cd preprocess/
###### python run.py --root_path {ROOT_PATH} --gender {GENDER}

## Avatar Training 
###### conda activate avatar_training
###### cd  avatar_training/
###### python main.py --base=./configs/gaussians_docker_custom.yaml
## Results
The visual results will be available in Results/SubjectID. 
To visualize the avatar and animate it, please move state.json and avatar.ply to training_asserts/ in Unity Editor package.
