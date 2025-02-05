# Fitting SMPL-X to a monocular video

Navigate to the Preprocessor folder first:
~~~
cd Preprocessor/
~~~

We are using the basic code structure from [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE/tree/main/fitting), and using models : [DECA](https://github.com/yfeng95/DECA), [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE), [mmpose](https://github.com/open-mmlab/mmpose). We also packaged complied models to a tools folder, which is uploaded on Hugging face https://huggingface.co/RendongZhang/GSAC-Dependencies/tree/main. For each download, run :
~~~
./download.sh
~~~

Then place your data you would like to process under data/Custom/data/{$SUBJECT_ID}.
Run :
download, run :
~~~
python run.py --root_path {PATH}/{$SUBJECT_ID} --gender {$GENDER}
~~~
Result will be placed in data/Custom/data/{$SUBJECT_ID}/CustomDataSet.



