import glob
from sklearn.model_selection import train_test_split

def file_path_save(): 
    filenames = [] 
    files = sorted(glob.glob("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/images/*.jpg"))
    print(len(files))
    train, val = train_test_split(
        files, test_size = 0.2, shuffle=True
    )
    print(len(train))
    print(len(val))
    for i in range(len(train)):
        with open("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/train.txt", 'a') as f:
            f.write(train[i] + "\n") 
    for i in range(len(val)):
        with open("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/val.txt", 'a') as f:
            f.write(val[i] + "\n") 

if __name__ == '__main__': file_path_save()