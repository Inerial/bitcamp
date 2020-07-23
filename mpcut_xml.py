import cv2
import os,shutil
from xml.etree.ElementTree import parse

# opencv의 함수인 VideoCapture 함수를 사용
main_file_path = 'D:/Study/videos/videos'       # 비디오 파일 위치
main_save_path = 'D:/Study/videos/capturess'   # 저장할 위치

count = 0
for m in os.listdir(main_file_path):
    for n in os.listdir(main_file_path + '/' + m):
        file_path = main_file_path + '/' + m + '/' + n
        file_list = os.listdir(file_path) # 동영상 파일이 들어가 있는 폴더 파일 리스트

        print(file_list)
        for i in file_list :
            if i[-3:] != 'mp4': # mp4가 아니라면 스킵
                continue
            path = file_path + '/' + i # 폴더의 각 파일에 대한 경로
            
            ## mp4이름으로 저장할 위치에 폴더 생성
            if os.path.isdir(main_save_path + '/' + i[:-4]):
                shutil.rmtree(main_save_path + '/' + i[:-4])
            os.mkdir(main_save_path +'/' + i[:-4])
        
            ## mp4와 이름이 같은 xml파일 가져오기
            file_xml = i[:-3] + 'xml'
            tree = parse(file_path + '/' + file_xml) # 파싱
            
            for action in tree.getroot().find("object").findall("action"): # object 안의 action들을 모두 가져와 for문
                action_name = action.find("actionname").text ## action 안의 actionname의 text를 가져옴 (kick 등등)

                ## action_name으로 폴더 생성
                if os.path.isdir(main_save_path + '/' + i[:-4] +'/' + action_name):
                    shutil.rmtree(main_save_path + '/' + i[:-4]+'/' + action_name)
                os.mkdir(main_save_path +'/' + i[:-4]+'/' + action_name)
                
                for frame in action.findall("frame"): # 모든 frame 가져와 for문
                    vidcap = cv2.VideoCapture(path)  ## 영상을 초기화

                    start = int(frame.find("start").text)
                    end = int(frame.find("end").text)

                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start) ## start값의 프레임으로 이동

                    ret = True
                    while(ret) :
                        ret, image = vidcap.read() # return 값과 image를 읽어온다
                        now = int(vidcap.get(1))
                        if ret is False:
                            print("**********************************************")
                            continue
                        if(now > end) :
                            break

                        if(now % 7 == 0) : # 7프레임 당 1프레임만 저장
                            print('Saved frame number :' + str(int(vidcap.get(1))))
                            cv2.imwrite(main_save_path + '/' + i[:-4] + '/' + action_name +'/' + action_name + '_frames%d.jpg' % count, image) # 새롭게 .jpg 파일로 저장
                            print('Saved frame%d.jpg' % count)
                            count += 1

                    vidcap.release()

print("캡쳐 완료")