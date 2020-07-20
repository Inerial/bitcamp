import cv2
import os,shutil

# opencv의 함수인 VideoCapture 함수를 사용
main_file_path = 'D:/Study/videos/videos'
for m in os.listdir(main_file_path):
    for n in os.listdir(main_file_path + '/' + m):
        file_path = main_file_path + '/' + m + '/' + n
        file_list = os.listdir(file_path) # 동영상 파일이 들어가 있는 폴더 파일 리스트

        print(file_list)
        count = 0
        for i in file_list :
            if i[-3:] != 'mp4':
                continue
            path = file_path + '/' + i # 폴더의 각 파일에 대한 경로

            if os.path.isdir('D:/Study/videos/capture/' + i[:-4]):
                shutil.rmtree('D:/Study/videos/capture/' + i[:-4])
            os.mkdir('D:/Study/videos/capture/' + i[:-4])
            vidcap = cv2.VideoCapture(path)

            ret = True
            while(ret) :
                ret, image = vidcap.read() # return 값과 image를 읽어온다
                if(ret == False) :
                    break

                if(int(vidcap.get(1)) % 5 == 0) : # 5프레임 당 1프레임만 저장
                    print('Saved frame number :' + str(int(vidcap.get(1))))
                    cv2.imwrite('D:/Study/videos/capture/' + i[:-4] + '/' + 'frame%d.jpg' % count, image) # 새롭게 .jpg 파일로 저장
                    print('Saved frame%d.jpg' % count)
                    count += 1

            vidcap.release()

print("캡쳐 완료")