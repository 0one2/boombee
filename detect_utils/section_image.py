import os
import numpy as np
import cv2


def compute_iou(boxA, boxB):
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def draw_bbox(image, bboxes, show_label=True):
    image_h, image_w = image.shape[0], image.shape[1]

    for i in range(bboxes.shape[0]):
        coor = bboxes[i]
        score = bboxes[i][4]

        fontScale = 0.4

        bbox_color = (255, 0, 0)
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%.2f' % (score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 4, lineType=cv2.LINE_AA)
    return image


def section_image(img_name, place, iou_threshold=0.25, input_size=(4000, 3000)):   # 0206
    img_name_only = img_name[:-4]
    image_folder = 'output/' + img_name_only +'/mergedtxt/'
    file_list = os.listdir(image_folder)
    file_list.remove(img_name_only + '_m_center.txt')
    file_list_num = []
    near_index_couple = []

    delete_bbox = []
    for file in file_list:
        file = file.replace(img_name_only + '_m_', '').replace('.txt', '')
        file_list_num.append([int(file[0:2]), int(file[2:4])])

    for i in range(len(file_list_num) - 1):
        for j in range(i + 1, len(file_list_num)):
            if (abs(file_list_num[j][0] - file_list_num[i][0]) <= 1) and (
                    abs(file_list_num[j][1] - file_list_num[i][1]) <= 1):
                near_index_couple.append((i, j))

    for couple in near_index_couple:
        filea = file_list[couple[0]]
        fileb = file_list[couple[1]]
        txta = np.loadtxt(os.path.join(image_folder, filea).replace("\\", "/"), dtype=np.int16).reshape((-1, 5))
        txtb = np.loadtxt(os.path.join(image_folder, fileb).replace("\\", "/"), dtype=np.int16).reshape((-1, 5))
        for i in range(txta.shape[0]):
            for j in range(txtb.shape[0]):
                if compute_iou(txta[i], txtb[j]) >= iou_threshold:
                    if txta[i][4] >= txtb[j][4]:
                        delete_bbox.append([couple[1], j])
                    else:
                        delete_bbox.append([couple[0], i])

    final_bbox = np.array([[0, 0, 0, 0, 0]], dtype=np.int16)
    for i, file in enumerate(file_list):
        txt = np.loadtxt(os.path.join(image_folder, file).replace("\\", "/"), dtype=np.int16).reshape((-1, 5))
        rows = []
        for index in delete_bbox:
            if index[0] == i:
                rows.append(index[1])
        txt = np.delete(txt, rows, axis=0)
        final_bbox = np.concatenate((final_bbox, txt), axis=0)
    final_bbox = np.delete(final_bbox, 0, axis=0)

    final_bbox = np.concatenate((final_bbox, np.loadtxt(os.path.join(image_folder, img_name_only + '_m_center.txt').replace("\\", "/"),
                                                        dtype=np.int16).reshape((-1, 5))), axis=0)

    #print(final_bbox)
    print(place)
    src= cv2.imread('output/image/'+place+'/'+img_name_only+'_resized.jpg', cv2.IMREAD_COLOR)
    src_bbox=draw_bbox(src,final_bbox,show_label=True)

    src_bbox=src_bbox[:input_size[1], :input_size[0]]      # 0206
    cv2.imwrite('output/image/'+place+'/'+img_name_only+'_final.jpg', src_bbox)

    np.savetxt('output/bbox/'+place+'/'+f'{img_name_only}_final.txt', final_bbox, fmt='%i')


