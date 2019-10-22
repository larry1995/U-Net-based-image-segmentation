# U-Net-based-image-segmentationUse python main.py to run the U-Net model
Data.py is the data generator and result saver 
Model.py stores the U-Net model.

Data folder: 


Train/image_ilker is the training dataset
Train/mask_ilker is the mask of the training dataset
Train/augment is the folder for augmented data produced during training
Train/training and training_groundtruth is for tif version, but not as input now.

Validation/vali_groundtruth is the mask of the validation dataset
Validation/vali is the validation dataset

For real test
Please use the images under public_test folder

