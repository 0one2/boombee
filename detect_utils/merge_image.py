import cv2
import os
import numpy as np
import warnings


def merge_image(img_name, max_people_size, input_size=(4000, 3000), output_size=(416, 416)):
    m = max_people_size
    n = []  # 가로,세로 타일 갯수
    resized_size = []  # resized size
    padding_size=[]   # 0206
    center_box = np.array([[0, 0, 0, 0, 0]], dtype=np.int16)

    for i, (inp, out) in enumerate(zip(input_size, output_size)):
        n.append((inp - m) // (out - m)+1)  # 0206
        resized_size.append(n[i] * (out - m) + m)
        padding_size.append(resized_size[i]-input_size[i])   # 0206

    img_name_only = img_name[:-4]
    # merge 된 resized 이미지 사이즈로 박스좌표 바꾸기


    for i in range(n[0]):  # ->진행 세로줄 격자
        for j in range(n[1]):  # 아래 진행 가로줄 격자
            filename =  f'{img_name_only}_{j:#02d}{i:#02d}.txt'
            path = 'output' + '/' + img_name_only + '/'
            section_path = path + 'rawtxt'

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                txt = np.loadtxt(os.path.join(section_path, filename).replace("\\", "/"), dtype=np.int16).reshape((-1, 5))

            if txt.shape[0] > 0:
                lefttop_x = (output_size[0] - m) * i
                lefttop_y = (output_size[1] - m) * j
                plus = np.array([lefttop_y, lefttop_x, lefttop_y, lefttop_x, 0])

                rows=[]
                for row in range(txt.shape[0]):  ## 겹치지 않는 영역 안에 있는지 없는지
                    if (txt[row][0] >= m) and (txt[row][1] >= m) \
                            and (txt[row][2] < (output_size[1] - m)) and (txt[row][3] < (output_size[0] - m)):
                        #print(txt[row])
                        center_box = np.concatenate((center_box, np.array([txt[row]+plus])), axis=0)
                        #print(center_box)
                        rows.append(row)
                    #else:
                        #print(txt[row], 'x')

                txt = np.delete(txt, rows, axis=0)  #center box 해당 rows 지우기

                if txt.shape[0] > 0:  #center box를 제거 후에도 남은 경우만
                    mapped_txt = txt + plus
                    mapped_filename = f'{img_name_only}_m_{j:#02d}{i:#02d}.txt'

                    os.makedirs(os.path.join(path, 'mergedtxt').replace("\\", "/"), exist_ok=True)
                    savepath = path + 'mergedtxt' + '/'
                    np.savetxt(os.path.join(savepath, mapped_filename).replace("\\", "/"), mapped_txt, fmt='%i')



            else:
                continue
    os.makedirs(os.path.join(path, 'mergedtxt').replace("\\", "/"), exist_ok=True)
    savepath = path + 'mergedtxt' + '/'
    center_box = np.delete(center_box,0,axis=0) #첫줄 지우기
    np.savetxt(os.path.join(savepath, f'{img_name_only}_m_center.txt').replace("\\", "/"), center_box, fmt='%i') ##hihi

