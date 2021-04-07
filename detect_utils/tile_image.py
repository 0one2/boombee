import cv2
import os


def tile_image(img_path, max_people_size, place, input_size=(4000, 3000), output_size=(416, 416)):
    m = max_people_size
    n = []  # 가로,세로 타일 갯수
    resized_size = []  # resized size
    padding_size=[]  # 0206 right,bottom padding_size

    for i, (inp, out) in enumerate(zip(input_size, output_size)):
        n.append((inp - m) // (out - m)+1)  # 0206
        resized_size.append(n[i] * (out - m) + m)
        padding_size.append(resized_size[i]-input_size[i])   # 0206

    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    dst = cv2.copyMakeBorder(src, 0, padding_size[1], 0, padding_size[0], cv2.BORDER_CONSTANT)  # 0206

    img_name = img_path.split('/')[-1]
    img_name_only = img_name[:-4]
    path = 'output/' + img_name_only + '/'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite('output/image/'+place+'/'+img_name_only+'_resized.jpg', dst)

    for i in range(n[0]):  # ->진행 세로줄 격자
        left = (output_size[0] - m) * i
        right = output_size[0] + (output_size[0] - m) * i
        for j in range(n[1]):  # 가로줄 격자
            top = (output_size[1] - m) * j
            bottom = output_size[1] + (output_size[0] - m) * j

            tile = dst[top:bottom, left:right]
            filename = f'{img_name_only}_{j:#02d}{i:#02d}.jpg'

            cv2.imwrite(os.path.join(path, filename).replace("\\", "/"), tile)



