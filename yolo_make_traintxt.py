import glob 
from sklearn.model_selection import train_test_split
def file_path_save(): 
    filenames = [] 
    files = sorted(glob.glob("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/images/*.jpg")) 
    train_files, val_files = train_test_split(
        files, shuffle=True, train_size=0.8, random_state=66
    )
    print(len(train_files))
    print(len(val_files))

    for i in range(len(train_files)): 
        f = open("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/train.txt", 'a') 
        f.write(train_files[i] + "\n") 

    for i in range(len(val_files)): 
        f = open("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/val.txt", 'a') 
        f.write(val_files[i] + "\n") 
        
if __name__ == '__main__': file_path_save()
