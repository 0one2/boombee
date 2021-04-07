from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.crowd_density import crowd_density
from detect_utils import tile_image, merge_image, section_image

import os
import sys
import time
import datetime
import argparse
import sys

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="data/50.JPG", help="name of image in data folder")
    parser.add_argument("--data_folder", type=str, default="detect_data", help="name of image in data folder")
    # parser.add_argument("--image_folder", type=str, default="output/50", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_198.pth", help="path to weights file")
    parser.add_argument("--pp_weights_path", type=str, default="pp_weights/best_weights.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.03, help="iou thresshold for non-maximum suppression on detection")
    parser.add_argument("--iou_thres", type=float, default=0.1, help="iou thresshold for merge image")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--people_pixel_size", type=int, default=30, help="size of people pixel size")
    parser.add_argument("--input_img_width", type=int, default=8000, help="size of each image dimension")
    parser.add_argument("--input_img_height", type=int, default=6000, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)

    # try:

    preprocessing = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Image Tiling End\n')
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    pp_model = classification().to(device)


    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path,map_location=device))

    pp_model.load_state_dict(torch.load(opt.pp_weights_path,map_location=device))

    model.eval()  # Set in evaluation mode
    pp_model.eval()

    square_pad = SquarePad()
    resize = transforms.Resize((112, 112))

    output_data = []

    for place in os.listdir(opt.data_folder):

        place_box_count = 0

        for image_name in os.listdir(os.path.join(opt.data_folder, place).replace("\\", "/")):

            box_count = 0

            image_path = os.path.join(opt.data_folder, place, image_name).replace("\\", "/")
            tmp_img = Image.open(image_path)
            meta_data = tmp_img._getexif()
            make_time = meta_data[36867]
            tmp_date = make_time.split(' ')[0].split(":")
            tmp_time = make_time.split(' ')[1]
            make_time = tmp_date[0] + "-" + tmp_date[1] + "-" + tmp_date[2] + " " + tmp_time

            tile_image.tile_image(image_path, opt.people_pixel_size, place, (opt.input_img_width, opt.input_img_height), (opt.img_size, opt.img_size))
            # image_name = opt.image_path.split('/')[-1]
            image_folder = 'output/'+image_name[:-4]
            os.makedirs("output", exist_ok=True)

            dataloader = DataLoader(
                ImageFolder(image_folder, transform= \
                    transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)])),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_cpu,
            )

            classes = load_classes(opt.class_path)  # Extracts class labels from file

            Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            imgs = []  # Stores image paths
            img_detections = []  # Stores detections for each image index

            print("\nPerforming object detection:")
            prev_time = time.time()
            for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
                # Configure input
                input_imgs = Variable(input_imgs.type(Tensor))

                # Get detections
                with torch.no_grad():
                    detections = model(input_imgs)
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

                # post processing with classification model
                pp_detections = []
                count = 0

                if detections != [None]:
                    for bboxes in detections:
                        for bbox_idx in range(bboxes.shape[0]):
                            bbox = bboxes[bbox_idx].long()
                            if bbox[0] < 0 :
                                bbox[0] = 0
                            if bbox[1] < 0 :
                                bbox[1] = 0
                            crop_images = input_imgs[:,:,bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1].clone()
                            crop_images = crop_images.squeeze(0)
                            crop_images = transforms.ToPILImage()(crop_images)
                            crop_images = preprocessing(crop_images).to(device)
                            for i in range(crop_images.shape[0]):
                                crop_images[i,:,:] = torch.transpose(crop_images[i,:,:],0,1)
                            pp_output = pp_model(crop_images.unsqueeze(0))
                            pp_output = F.softmax(pp_output,dim=1)
                            if pp_output[0,1] > 0.41:
                                pp_class = 1
                            else :
                                pp_class = 0
                            # _, pp_class = torch.max(pp_output, 1)

                            if pp_class == 1:
                                if pp_detections == []:
                                    pp_detections.append(bboxes[bbox_idx].unsqueeze(0))
                                    count += 1
                                else :
                                    pp_detections[0] = torch.cat((pp_detections[0],bboxes[bbox_idx].unsqueeze(0)),dim=0)
                                    count += 1

                if pp_detections == []:
                    pp_detections=[None]

                detections = pp_detections


                # Log progress
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                prev_time = current_time
                print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

                # Save image and detections
                imgs.extend(img_paths)
                img_detections.extend(detections)
            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, 20)]

            print("\nSaving images:")
            # Iterate through images and save plot of detections
            for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

                print("(%d) Image: '%s'" % (img_i, path))

                # Create plot
                img = np.array(Image.open(path))
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                box_loc_conf = []  ##

                # Draw bounding boxes and labels of detections
                if detections is not None:
                    print(detections)
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)


                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        box_count += 1

                        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                        row=list(map(round,[y1.item(),x1.item(),y2.item(),x2.item(),conf.item()*10000]))
                        box_loc_conf.append(row)  ##

                        box_w = x2 - x1
                        box_h = y2 - y1

                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # Add label
                        plt.text(
                            x1,
                            y1,
                            s=classes[int(cls_pred)],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                        )

                boxes_loc_conf = np.array(box_loc_conf)      ##

                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                filename = os.path.basename(path).split(".")[0]
                print(filename)
                print(boxes_loc_conf)

                output_path = os.path.join(image_folder, f"{filename}.png").replace("\\", "/")
                os.makedirs(image_folder+'/rawtxt', exist_ok=True)
                np.savetxt(os.path.join(image_folder+'/rawtxt', f"{filename}.txt").replace("\\", "/"), boxes_loc_conf, fmt='%i')      ##
                plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
                plt.close()

            merge_image.merge_image(image_name, opt.people_pixel_size, (opt.input_img_width, opt.input_img_height), (opt.img_size, opt.img_size))
            # section_image.section_image(image_name, place, opt.iou_thres, (opt.input_img_width, opt.input_img_height))  # 0206
            place_box_count += box_count
        output_data.append([place,make_time,place_box_count])
        print(output_data)


    area_data = pd.read_csv("data/park_area.csv")
    crowd_density(output_data,area_data)
    #     sys.exit(1)
    #
    # except:
    #     sys.exit(0)

    
