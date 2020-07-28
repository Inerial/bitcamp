from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            validation_split=0.2
        )

# img = load_img('D:/Study/data/dog_cat/cat.jpg')  # PIL 이미지
# x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
# x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

i = 0

for batch in datagen.flow_from_directory(
            directory='D:/Study/data/dog_cat',
            shuffle=False,
            batch_size=1,
            subset='training',
            save_to_dir='D:/Study/data/preview'):
            i+=1
            print(batch)
            if i is 30:
                break


# # 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
# # 지정된 `preview/` 폴더에 저장합니다.
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='D:/Study/data/dog_cat/preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # 이미지 20장을 생성하고 마칩니다