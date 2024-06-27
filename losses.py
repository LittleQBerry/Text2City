from torch.nn import functional as F
import torch.nn as nn
import torch
from augmentations import ImageAugmentations
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
import cv2
from sold.misc.visualize_util import plot_lines

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    #print(x.shape)
    #print(y.shape)
    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    #print("d",distance)
    return distance


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def reconstruct_loss(x, y, mask, th):
    l1_loss = nn.L1Loss()
    loss_valid = l1_loss(x*(1-mask),y* (1-mask))
    loss_all =loss_valid
    return loss_all

def clip_loss(x_in, text_embed,mask,clip_model,batch_size,image_augmentations,clip_normalize):
    clip_loss = torch.tensor(0)

    if mask is not None:
        masked_input = x_in * mask
    else:
        masked_input = x_in
    augmented_input = image_augmentations(masked_input).add(1).div(2)
    clip_in = clip_normalize(augmented_input)
    image_embeds = clip_model.encode_image(clip_in).float()
    dists = d_clip_loss(image_embeds, text_embed)

    for i in range(batch_size):
        clip_loss = clip_loss + dists[i :: batch_size].mean()

    return clip_loss


def EdgeLoss(input,gt, model ,mask):
    l1_loss=nn.L1Loss()
    edge_loss= torch.tensor(0)
    generated =input[0].add(1).div(2).clamp(0,1)
    gt_i =gt[0].add(1).div(2)
    input_c_hole =model(generated*mask[0])
    gt_hole =model(gt_i*mask[0])
    edge_loss =edge_loss+l1_loss(input_c_hole,gt_hole)
    print(edge_loss)
    return edge_loss


def footprint2(input, model):
    generated =input[0].add(1).div(2).clamp(0,1)
    input_c_hole = model(generated)
    print(input_c_hole.shape)
    return input_c_hole

def extract_line(image,model):
    loader = transforms.Compose(
        [transforms.ToTensor()]
    )

    image = image[0].add(1).div(2).clamp(0,1)
    devices =image.device
    #print(image.shape)
    image = TF.to_pil_image(image)
    image_gray = TF.to_grayscale(image)
    #image = Image.fromarray(image.cpu().numpy())
    #image_gray =image.convert("L")
    image_gray =loader(image_gray).to(devices)
    #print(image.shape)
    #print(image_gray.shape)
    image_gray =image_gray.unsqueeze(0)
    #print(image_gray.shape)
    #exit()
    #print(image.shape)
    
    ref_heatmap =model.line_detection(image_gray)["heatmap"]
    ref_heatmap =np.where(ref_heatmap<0.6,0,1)
    ref_heatmap2 =np.where(ref_heatmap<0.9,0,1)
    

    #ref_line_seg = model.line_detection(image_gray)['line_segments']
    
    
    #print(ref_line_seg)
    #exit()
    #print(ref_heatmap.type)
    #exit()
    '''
    ref_line_seg =model.line_detection(image_gray)["line_segments"]
    images =Image.new("L",[256,256],"black")
    imgs =ImageDraw.Draw(images)
    print(ref_line_seg)
    for i in range(len(ref_line_seg)):
        imgs.line([(ref_line_seg[i, 0, 0], ref_line_seg[i, 1, 0]),
                (ref_line_seg[i, 0, 1], ref_line_seg[i, 1, 1])],
                fill="white",width=4
                )

    ref_heatmap = np.array(images)
    ref_heatmap2 = np.array(images)
    '''           
    #print(ref_line_seg.shape)
    #exit()
    #return loader(images).to(devices)
    return ref_heatmap,ref_heatmap2


def footprint(image):
    loader = transforms.Compose(
        [transforms.ToTensor()]
    )
    image = image[0].add(1).div(2).clamp(0,1)
    devices =image.device
    image = TF.to_pil_image(image)
    image_gray = TF.to_grayscale(image)
    image_gray = np.array(image_gray)
    blur =cv2.blur(image_gray,(9,9))
    medianblur =cv2.medianBlur(image_gray,9)

    ret,img_bin = cv2.threshold(medianblur,182,255,cv2.THRESH_BINARY)

    contour, hierarchy =cv2.findContours(img_bin,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    draw_img = medianblur.copy()
    rets =cv2.drawContours(draw_img, contour,-1, 255,2)

    return img_bin /255

NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]

def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points

def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img

def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img

def get_compactness_cost(y_pred):
    
    #print(y_pred.shape)
    
    y_pred = y_pred[:,0,:,:]
    x = y_pred[:,1:,:] - y_pred[:,:-1,:]
    y = y_pred[:,:,1:] - y_pred[:,:,:-1]
    #x = y_pred
    #print(x.shape)
    #exit()
    #print(x)
    delta_x = x[:, :, 1:] ** 2
    delta_y = y[:, 1:, :] ** 2
    #print()
    delta_u = torch.abs(delta_x + delta_y)
    epsilon = 0.00000001 
    w = 0.01
    length = w * torch.sum(torch.sqrt(delta_u), dim=[1, 2]) + epsilon
    
    
    area = torch.sum(y_pred, dim=[1, 2]) +epsilon
    
    compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.14))
    #if compactness_loss >1000:
        #print('length',length)
        #print('y_pred',y_pred.shape)
        #print('area',area)
        #img_grid = [y_pred[0].reshape(1,512,512),y_pred[1].reshape(1,512,512),y_pred[2].reshape(1,512,512),y_pred[3]]
        #save_image(img_grid,'test.png')
        #exit()
    #print(compactness_loss.shape)
    #print(compactness_loss)
    #print(length)
    #print(area)
    #exit()

    return compactness_loss

#def 
'''
class EdgeLoss(nn.Module):
    def __init__(self, cannynet):
        super().__init__()
        self.l1 =nn.L1Loss()
        self.cannynet=cannynet

    def forward(self, input, gt):
        #print("i")
        #print(input.shape)
        #print(output.shape)
        #exit()
        #loss_e =0.0
        loss_e_hole=torch.tensor(0)
        for i in range(input.shape[0]):
            #print(input[i])
            #print(output[i])
            #if torch.isnan(output[i]).all():
                #loss_e =loss_e+10.0
            #    loss_e = torch.tensor(loss_e+1.0).cuda()
                #print("nan")
            #else:
                #input_c =self.cannynet(input[i])
                #output_c = self.cannynet(output[i])
                #print("i",input[i].shape)
            input_c_hole =self.cannynet(input[i])
            gt_hole =self.cannynet(gt[i])
            #output_c_hole = self.cannynet(output[i]*(1-mask[i])+input[i]*mask[i])
            
            #imsave('final.png', (output_c_hole.data.cpu().numpy()[0, 0] > 0.0).astype(float))
            #thresholded.data.cpu().numpy()[0, 0] > 0.0
            #loss_e =loss_e+self.l1(output_c,input_c)
            loss_e_hole =loss_e_hole+self.l1(input_c_hole,gt_hole)

        return loss_e_hole
'''
