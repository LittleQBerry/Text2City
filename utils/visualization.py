from pathlib import Path
from numpy.core.shape_base import block
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image
from PIL import Image as im
import cv2
from losses import d_clip_loss, range_loss, reconstruct_loss, clip_loss,EdgeLoss,extract_line,footprint2, Two_Pass,footprint
from skimage import morphology
from torchvision import transforms

def show_tensor_image(tensor: torch.Tensor, range_zero_one: bool = False):
    """Show a tensor of an image

    Args:
        tensor (torch.Tensor): Tensor of shape [N, 3, H, W] in range [-1, 1] or in range [0, 1]
    """
    if not range_zero_one:
        tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)

    batch_size = tensor.shape[0]
    for i in range(batch_size):
        plt.title(f"Fig_{i}")
        pil_image = TF.to_pil_image(tensor[i])
        plt.imshow(pil_image)
        plt.show(block=True)

def save_editied_masked_image(
    title,
    p,
    source_image,
    edited_image,
    source_h,
    edited_h,
    source_b,
    edited_b: Image,
    mask: Optional[Image] = None,
    path: Optional[Union[str, Path]] = None,
    #p: Optional[Union[str, Path]] = None,
    distance: Optional[str] = None,
):
    
    fig_idx = 1
    rows = 4
    cols = 2 if mask is not None else 2

    fig = plt.figure(figsize=(12, 36))
    figure_title = f'Prompt: "{title}"'
    if distance is not None:
        figure_title += f" ({distance})"
    plt.title(figure_title)
    plt.axis("off")

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source Image")
    plt.imshow(source_image)



    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Edited Image")
    plt.imshow(edited_image)

    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source highway")
    plt.imshow(source_h)

    
    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Regenerated highway")
    plt.imshow(edited_h)

    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source building")
    plt.imshow(source_b)

    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Regenerated building")
    plt.imshow(edited_b)


    fig.add_subplot(rows, cols, fig_idx)
    fig_idx +=1
    _set_image_plot_name("Mask")
    plt.imshow(mask)

    fig.add_subplot(rows, cols, fig_idx)
    _set_image_plot_name("Results")
    
    
    
    
    #denoise
    edited_b =TF.to_grayscale(edited_b)
    mask =TF.to_grayscale(mask)
    edited_b =np.array(edited_b)
    mask =np.array(mask)
    mask =mask/255
    print(mask)
    edited_b_m = edited_b *mask
    #edited_b_m =TF.to_grayscale(edited_b_m)
    
    #edited_b_m =np.array(edited_b_m)
     
    ret,img_bin =cv2.threshold(edited_b_m,200,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5),np.uint8)

    edited_b_m = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN,kernel)
    edited_b_b = edited_b*(1-mask) +edited_b_m *mask
    edited_b_b = np.dstack((edited_b_b,edited_b_b,edited_b_b))



    new_img = np.where(edited_b_b<220,edited_h,edited_b_b)
    new_img = im.fromarray(new_img.astype('uint8')).convert('RGB')

    
    plt.imshow(new_img)
    plt.gray()
    new_img.save(path)
    #exit()

def show_editied_masked_image(
    title: str,
    p: str,
    source_image: Image,
    edited_image: Image,
    source_h :Image,
    edited_h: Image,
    source_b: Image,
    edited_b: Image,
    mask: Optional[Image] = None,
    path: Optional[Union[str, Path]] = None,
    #p: Optional[Union[str, Path]] = None,
    distance: Optional[str] = None,
):
    path
    fig_idx = 1
    rows = 4
    cols = 2 if mask is not None else 2

    fig = plt.figure(figsize=(12, 36))
    figure_title = f'Prompt: "{title}"'
    if distance is not None:
        figure_title += f" ({distance})"
    plt.title(figure_title)
    plt.axis("off")

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source Image")
    plt.imshow(source_image)



    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Edited Image")
    plt.imshow(edited_image)

    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source highway")
    plt.imshow(source_h)

    
    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Regenerated highway")
    plt.imshow(edited_h)


    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source building")
    plt.imshow(source_b)

    fig.add_subplot(rows,cols,fig_idx)
    fig_idx += 1
    _set_image_plot_name("Regenerated building")
    plt.imshow(edited_b)

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx +=1
    _set_image_plot_name("Mask")
    plt.imshow(mask)

    fig.add_subplot(rows, cols, fig_idx)
    _set_image_plot_name("Results")
    
    
    
    
    #denoise
    edited_b =TF.to_grayscale(edited_b)
    mask =TF.to_grayscale(mask)
    edited_b =np.array(edited_b)
    mask =np.array(mask)
    mask =mask/255
    print(np.max(mask))
    edited_b_m = edited_b *mask

     
    ret,img_bin =cv2.threshold(edited_b_m,100,255,cv2.THRESH_BINARY)
    
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    edited_b_m = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN,kernel)
    edited_b_b = edited_b*(1-mask) +edited_b_m *mask
    ret, img_m = cv2.threshold(edited_b_b,200,255,cv2.THRESH_BINARY)  
    img_m_o = img_m*(1-mask)

    print(np.max(img_m))
    img_s= morphology.remove_small_objects((img_m/255).astype(bool),min_size=300,connectivity=1)
    img_s = img_s *mask +img_m_o

    building_mask = img_m/255
    building_mask =building_mask.reshape(256,256,1)
    edited_h =np.array(edited_h)
    new_img = edited_h * (1-building_mask) + np.dstack((img_m,img_m,img_m))
    #new_img = np.where(edited_b_b<220,edited_h,edited_b_b)
    new_img = im.fromarray(new_img.astype('uint8')).convert('RGB')

    path4 = "/{}_a.png".format(p)
    new_img.save(path4)

    '''
    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show(block=True)
    '''
    plt.close()



def show_result(m,gt_a,gt_h,gt_b,r_h,r_b,p):
    #image
    mask = m.cpu()
    gt_a = gt_a.add(1).div(2).cpu()
    gt_h = gt_h.add(1).div(2).cpu()
    gt_b = gt_b.add(1).div(2).cpu()
    r_h = r_h.cpu()
    r_b = r_b.cpu()

    # tensor -->numpy
    mask  = mask.numpy()
    gt_a  = gt_a.numpy() *255
    gt_h  = gt_h.numpy() *255
    gt_b = gt_b.numpy() *255
    r_h = r_h.numpy() *255
    r_b = r_b.numpy()*255
    
    mask =mask.reshape(mask.shape[1],mask.shape[2],mask.shape[0])
    gt_a = gt_a.reshape(gt_a.shape[1],gt_a.shape[2],gt_a.shape[0])
    gt_h = gt_h.reshape(gt_h.shape[1],gt_h.shape[2],gt_h.shape[0])
    gt_b = gt_b.reshape(gt_b.shape[1],gt_b.shape[2],gt_b.shape[0])
    r_h = r_h.reshape(r_h.shape[1],r_h.shape[2],r_h.shape[0])
    r_b = r_b.reshape(r_b.shape[1],r_b.shape[2],r_b.shape[0])


    #building
    r_b_m  = r_b * mask 

    ret, img_bin = cv2.threshold(r_b_m , 100,255, cv2.THRESH_BINARY)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    r_b_m = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN,kernel)
    

    r_b_new = r_b_m * mask
    
    #road
    r_h_new = r_h *mask

    result = gt_a * (1-mask) +r_h *mask +r_b_new *mask
    result = result + r_b_m
    
    print(result.shape)

    new_image = im.fromarray(result.astype('uint8')).convert('RGB')
    path = "/{}_a.png".format(p)
    new_image.save(path)

def save_pil(m,gt_a,r_h,r_b,p,line_matcher):
    #load img
    back_ground = im.open("/home3/qinyiming/work2/ours/background.png").convert('RGB')
    back_ground = np.array(back_ground)
    road = im.open('/home3/qinyiming/work2/ours/road.png').convert('RGB')
    road = np.array(road)

    path_h = "/{}_h.png".format(p)
    
    path_b = "//{}_b.png".format(p)
    #r_b.save(path_b)
    mask = TF.to_grayscale(m)
    mask = np.array(mask)
    mask = mask/255

    gt_a = np.array(gt_a)
    #gt_a = TF.to_grayscale(gt_a)

    r_b =TF.to_grayscale(r_b)
    r_b = np.array(r_b)
    #print('r_b',r_b)
    
    #medianBlur
    
    
    r_h = TF.to_grayscale(r_h)
    r_h.save(path_h)
    loader = transforms.Compose(
       [transforms.ToTensor()]
    )
    ##r_h = loader(r_h).to('cuda:1')
    #r_h = np.array(r_h)
    #print('r_h',r_h)


    
    #building part
    r_b_m = r_b * mask

    ret, img_bin = cv2.threshold(r_b_m,230,255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN,(3,3))
    r_b_m = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN, kernel)

    r_b_new = np.dstack((r_b_m,r_b_m,r_b_m))
    r_b_im = im.fromarray(r_b_new.astype('uint8')).convert('RGB')
    r_b_im.save(path_b)
    
    
    #road 

    
    # bilateralFilter
    r_h =np.array(r_h)
    r_h = cv2.bilateralFilter(r_h,3,75,75)
    r_h = np.where(r_h<170,255.0,0.0)
    road_mask_1 = (r_h *mask)/255
    save_r_h_1 = r_h.astype('uint8')
    path_1 = "/{}_sr.png".format(p)
    cv2.imwrite(path_1,save_r_h_1 )
    #line-detection
    r_h = im.open(path_1)
    r_h = TF.to_grayscale(r_h)
    r_h = loader(r_h).to('cuda:1')
    #r_h = torch.tensor(r_h.astype('float32')).to('cuda:1')
    r_h =r_h.unsqueeze(0)
    ref_heatmap = line_matcher.line_detection(r_h)['heatmap']
    ref_heatmap = np.where(ref_heatmap<1.0,0,1)
    r_h_mask =ref_heatmap * mask
    #save_r_h_2 =(r_h_mask*255).astype('uint8')
    #path_2 = "/r/{}_ld.png".format(p)
    #cv2.imwrite(path_2, save_r_h_2)

    #new road mask
    road_mask = road_mask_1#+ r_h_mask
    road_mask = np.where(road_mask==2,1,road_mask)
    save_r_h_2 =(road_mask*255).astype('uint8')
    path_2 = "//{}_ld.png".format(p)
    cv2.imwrite(path_2, save_r_h_2)
    
    #calculate new mask
    #r_h_mask is road mask
    #r_b_m is building mask

    #mask is original mask
    new_mask = mask - (road_mask)
    builing_mask = r_b_m /255
    new_mask = new_mask - (builing_mask)
    new_mask = np.where(new_mask<0,0,new_mask)
    builing_mask = builing_mask - road_mask
    builing_mask = np.where(builing_mask<0,0,builing_mask)

    #new background
    new_bg = back_ground * np.dstack((new_mask,new_mask,new_mask))
    new_road = road * np.dstack((road_mask,road_mask,road_mask))
    new_building = np.dstack((builing_mask,builing_mask,builing_mask))* 255
    #output
    output_h =im.fromarray(new_road.astype('uint8')).convert('RGB')
    #output_h.save(path_h)
    #all
    c_mask  = np.dstack((1-mask,1-mask,1-mask))
    result = gt_a * c_mask + new_bg + new_road + new_building
    #result = gt_a * c_mask + 
    new_image = im.fromarray(result.astype('uint8')).convert('RGB')
    path = "/{}_a.png".format(p)
    new_image.save(path)
    








    

def _set_image_plot_name(name):
    plt.title(name)
    plt.xticks([])
    plt.yticks([])
