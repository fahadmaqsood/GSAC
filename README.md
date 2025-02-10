# GSAC: Leveraging Gaussian Splatting for Photorealistic Avatar Creation with Unity Integration
Gaussian Splatting Avatar Creation Release 

This repo releases our end-to-end pipeline for gaussian splatting avatar creation for a monocular video in ~ 40 mins. Our pipeline incorporates a novel Gaussian splatting technique with
customized preprocessing, enabling detailed facial expression reconstruction and seamless integration with Unity-based VR/AR platforms. Additionally, we present a Unity-integrated Gaussian
Splatting Avatar Editor, offering a user-friendly environment for VR/AR application development. 
## Install
Clone the repo
~~~
git clone https://github.com/VU-RASL/GSAC.git
~~~
cd to the repo, then run
~~~
bash Avatar/docker build.sh
bash Avatar/docker run.sh
bash Avatar gaussian_install.sh
~~~
We recommand use docker image.
## Data Gathering 
Please record the video by rotating the yourself slowly and keeping the camera stable. The time would be about 20 secs. Please make sure there is only one human inside a video each of the time.
Place the recorded video under inputs folder:
~~~

ROOT
    |__Data/
        |__{SUBJECT}/frames/
    |__Preprocessor/
    |__Avatar/

~~~
## Simplest Run
We provide a single bash script that you can easily run the pipeline end-to-end easily. After docker is running and the data is pleased in correct place, navigate to ROOT folder of GSAC, run the command below:
~~~
bash create_avatar.sh {SUBJECT} {GENDER}
~~~


Othersiwe you can run the preprocessing and training separately.
### Data Preprocessing 
For capturing SMPLX parameters from video frames, inside the docker environment we created above, please run our preprocessing using commands below （Note: for infant mode, gender is INFANT）:
~~~
cd Preprocessor
python run.py --root_path {SUBJECT} --gender {GENDER}
~~~
The fitted results will be placed in train_data under {Subject} folder :
~~~

ROOT
    |__Data/
        |__{SUBJECT}/
            |__frames/
            |__train/
    |__Preprocessor/
    |__Avatar/

~~~
### Avatar Training 
~~~
conda activate avatar_training
cd  avatar_training/
python main.py --base=./configs/gaussians_docker_custom.yaml
~~~
## Results
The visual results will be available in Results/SubjectID. 
To visualize the avatar and animate it, please move state.json and avatar.ply to training_asserts/ in Unity Editor package.
