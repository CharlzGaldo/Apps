Steps on how to run my app: (Watch the end of the YouTube video)  https://youtu.be/T15ZbdtPAzc
1. Download the zip files from GitHub [https://github.com/CharlzGaldo/Apps.git]. 

2. Download the DATASET from  [https://www.kaggle.com/datasets/harshitkandoi7850/dataset-for-visual-plastic-type-recognition?select=Plastic+types]  
(This is a test dataset you can use which is public, I used WADABA which is private to make the “CharlzModel.pt”) 

3. Extract the folder  

4. Run setup.bat. [Make sure you have python 3.12 installed before running the bat.] 
This bat file will create a venv (virtual environment) that has all the requirements to run the app.py 

4.1. If your computer does not support CUDA, please select “[1] CPU Only”.  
Note: Selecting and installing “[2] GPU (Cuda)” will take a lot longer to install and will not work if the machine does not support CUDA. 

5. Once the installer completes, close the .bat file and open the “run.bat” file. (Wait.) 

6. Once the app finally opens then it should be pretty self-explanatory from there. 

 
 
Testing the existing model: 

I have included in the github files “CharlzModel.pt” which is the trained model on the WADABA dataset so you can test it on your own images or improve the model. 
