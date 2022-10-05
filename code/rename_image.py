import os

listdir = r"E:\SHU\10901\AI Introduction\Final project\animals_recognize\Kaggle_Data\animals_recognize\123\Squirrel"
basedir = r"E:\SHU\10901\AI Introduction\Final project\animals_recognize\Kaggle_Data\animals_recognize\123\Squirrel\\"
movdir = r"E:\SHU\10901\AI Introduction\Final project\animals_recognize\Kaggle_Data\animals_recognize\test\\"

for count, filename in enumerate(os.listdir(basedir)): 
        dstfilename = str(25704+((count)*2)) + ".jpg"
        
        src = basedir + filename
        dst = movdir + dstfilename

        # rename() function will 
        # rename all the files 
        os.rename(src, dst)
