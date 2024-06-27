from asyncore import file_dispatcher
from email.mime import image
import json
import os
from PIL import Image, ImageDraw
import numpy as np
def visual_data(path):
    file_names= os.path.listdir()


def retrive_data(path):
    file_list=os.listdir(path)

    for file_ in file_list:
        name,_ = os.path.splitext(file_)
        file_name  =os.path.join(path, file_)
        print(file_)
        with open(file_name) as f :
            gj =json.load(f)
        coor_base =file_.split(', ')
        south=coor_base[0]
        west=coor_base[1]
        north=coor_base[2]
        east=coor_base[3][:-8]
        json_func={'coor_base':[south,west,north,east],'label':[],'data':[]}
        json_building={'coor_base':[south,west,north,east],'label':[],'data':[]}
        json_highway={'coor_base':[south,west,north,east],'label':[],'data':[]}
        json_railway={'coor_base':[south,west,north,east],'label':[],'data':[]}

        for i in range(len(gj['features'])):
            flag=True
            if gj['features'][i]['geometry']['type'] =='Point':
                print('point')
            
            if gj['features'][i]['geometry']['type'] =='LineString':
                if 'highway' in gj['features'][i]['properties']:
                    json_highway['label'].append(gj['features'][i]['properties']['highway'])
                    json_highway['data'].append(gj['features'][i]['geometry']['coordinates'])
                if 'railway' in gj['features'][i]['properties']:
                    json_railway['label'].append(gj['features'][i]['properties']['railway'])
                    json_railway['data'].append(gj['features'][i]['geometry']['coordinates'])
            if gj['features'][i]['geometry']['type'] =='MultiLineString':
                if 'highway' in gj['features'][i]['properties']:
                    for d in gj['features'][i]['geometry']['coordinates']:

                        json_highway['label'].append(gj['features'][i]['properties']['highway'])
                        json_highway['data'].append(d)
                if 'railway' in gj['features'][i]['properties']:
                    for d in gj['features'][i]['geometry']['coordinates']:
                        json_railway['label'].append(gj['features'][i]['properties']['railway'])
                        json_railway['data'].append(d)


            if gj['features'][i]['geometry']['type'] =='Polygon':
                if 'building' in gj['features'][i]['properties']:
                    
                    json_building['data'].append(gj['features'][i]['geometry']['coordinates'])
                    if gj['features'][i]['properties']['building']=='yes':
                        if 'leisure' in gj['features'][i]['properties']:
                            json_building['label'].append(gj['features'][i]['properties']['leisure'])
                        elif 'amenity' in gj['features'][i]['properties']:
                            json_building['label'].append(gj['features'][i]['properties']['amenity'])
                        elif 'natural' in gj['features'][i]['properties']:
                            json_building['label'].append(gj['features'][i]['properties']['natural'])
                        else:
                            json_building['label'].append(gj['features'][i]['properties']['building'])
                    else:
                        json_building['label'].append(gj['features'][i]['properties']['building'])
                    
                else:
                    if 'landuse' in gj['features'][i]['properties']:
                        json_func['label'].append(gj['features'][i]['properties']['landuse'])
                        json_func['data'].append(gj['features'][i]['geometry']['coordinates'])
                    elif 'leisure' in gj['features'][i]['properties']:
                        json_func['label'].append(gj['features'][i]['properties']['leisure'])
                        json_func['data'].append(gj['features'][i]['geometry']['coordinates'])
                    elif 'amenity' in gj['features'][i]['properties']:
                        json_func['label'].append(gj['features'][i]['properties']['amenity'])
                        json_func['data'].append(gj['features'][i]['geometry']['coordinates'])
                    elif 'natural' in gj['features'][i]['properties']:
                        json_func['label'].append(gj['features'][i]['properties']['natural'])
                        json_func['data'].append(gj['features'][i]['geometry']['coordinates'])
                    else:
                        print('not a building or landuse',file_name)
            
            if gj['features'][i]['geometry']['type'] =='MultiPolygon':
                if 'building' in gj['features'][i]['properties']:
                    for d in gj['features'][i]['geometry']['coordinates']:
                        
                        json_building['data'].append(d)
                        if gj['features'][i]['properties']['building']=='yes':
                            if 'leisure' in gj['features'][i]['properties']:
                                json_building['label'].append(gj['features'][i]['properties']['leisure'])
                            elif 'amenity' in gj['features'][i]['properties']:
                                json_building['label'].append(gj['features'][i]['properties']['amenity'])
                            elif 'natural' in gj['features'][i]['properties']:
                                json_building['label'].append(gj['features'][i]['properties']['natural'])
                            else:
                                json_building['label'].append(gj['features'][i]['properties']['building'])

                        else:
                            json_building['label'].append(gj['features'][i]['properties']['building'])

                else:

                    if 'landuse' in gj['features'][i]['properties']:
                        for d in gj['features'][i]['geometry']['coordinates']:
                            json_func['label'].append(gj['features'][i]['properties']['landuse'])
                            json_func['data'].append(d)
                    elif 'leisure' in gj['features'][i]['properties']:
                        for d in gj['features'][i]['geometry']['coordinates']:
                            json_func['label'].append(gj['features'][i]['properties']['leisure'])
                            json_func['data'].append(d)
                            
                    elif 'amenity' in gj['features'][i]['properties']:
                        for d in gj['features'][i]['geometry']['coordinates']:
                            json_func['label'].append(gj['features'][i]['properties']['amenity'])
                            json_func['data'].append(d)
                    elif 'natural' in gj['features'][i]['properties']:
                        for d in gj['features'][i]['geometry']['coordinates']:
                            json_func['label'].append(gj['features'][i]['properties']['natural'])
                            json_func['data'].append(d)
                    else:
                        print("not building or landuse",file_name)


        json_data1 =json.dumps(json_func,indent=4,separators=(',', ': '))
        f=open('/data/work2/retrived_data/{}_func.json'.format(name),'w')
        f.write(json_data1)
        f.close()

        json_data2 =json.dumps(json_highway,indent=4,separators=(', ',': '))
        f=open('/data/work2/retrived_data/{}_highway.json'.format(name),'w')
        f.write(json_data2)
        f.close()

        json_data3 =json.dumps(json_building,indent=4,separators=(', ',': '))
        f=open('/data/work2/retrived_data/{}_building.json'.format(name),'w')
        f.write(json_data3)
        f.close()

        json_data4=json.dumps(json_railway,indent=4,separators=(', ',': '))
        f=open('//data/work2/retrived_data/{}_railway.json'.format(name),'w')
        f.write(json_data4)
        f.close()

def translate(base,new,xy):
    new_X = ((xy[0]-float(base[1]))*new[1]+(float(base[3])-xy[0])*new[0])/(float(base[3])-float(base[1]))
    new_Y = ((xy[1]-float(base[2]))*new[1]+(float(base[0])-xy[1])*new[0])/(float(base[0])-float(base[2]))

    return [int(new_X),int(new_Y)]


def transfer_data(path):
    file_list = os.listdir(path)
    new=[0,512]
    for file_ in file_list:
        file_name =os.path.join(path,file_)

        with open(file_name) as f :
            gj =json.load(f)
        
        coor_base = gj['coor_base']
        
        for i in range(len(gj['label'])):
            data = gj['data'][i]
            
            for j in range(len(data)):
                if len(data[j])==2:
                    if type(data[j][0])==float:
                        print('1')
                        tmp =translate(coor_base,new,data[j])
                        data[j]=tmp
                    else:
                        print('2')
                        for m in range(len(data[j])):
                            tmp =translate(coor_base,new,data[j][m])
                            data[j][m]=tmp
                else:
                    print(3,file_name)
                    for m in range(len(data[j])):
                        tmp =translate(coor_base,new,data[j][m])
                        data[j][m]=tmp
                #mp=translatet(coor_base,new,data[j])
            gj['data'][i]=data
        
        gj['coor_base'][0]=new[0]
        gj['coor_base'][1]=new[1]
        gj['coor_base'][2]=new[1]
        gj['coor_base'][3]=new[0]

            
        jsondata6= json.dumps(gj,indent=4,separators=(',',':'))
        f_ = open('/data/work2/trans_data/{}'.format(file_),'w')
        f_.write(jsondata6)
        f_.close()

def count_label(path):
    file_list =os.listdir(path)
    dict1={}
    dict2={}
    dict3={}
    dict4={}
    for file in file_list:
        name,_ = os.path.splitext(file)
        file_name =os.path.join(path,file)
        with open(file_name) as f :
            gj =json.load(f)
        if name.split("_")[1]=='func':
            for i in range(len(gj['label'])):
                label = gj['label'][i]
                if label in dict1:
                    dict1[label] =dict1[label]+1
                else:
                    dict1[label]=1
        if name.split("_")[1]=='building':
            for i in range(len(gj['label'])):
                label = gj['label'][i]
                if label in dict2:
                    dict2[label] =dict2[label]+1
                else:
                    dict2[label]=1
        
        if name.split('_')[1]=='highway':
            for i in range(len(gj['label'])):
                label = gj['label'][i]
                if label in dict3:
                    dict3[label] =dict3[label]+1
                else:
                    dict3[label]=1
        if name.split("_")[1]=="railway":
            for i in range(len(gj['label'])):
                label = gj['label'][i]
                if label in dict4:
                    dict4[label] =dict4[label]+1
                else:
                    dict4[label]=1

    jsondata1 = json.dumps(dict1,indent=4,separators=(',',':'))
    f_ =open("func.json",'w')
    f_.write(jsondata1)
    f_.close()    

    jsondata2 =json.dumps(dict2,indent=4,separators=(',',':'))
    f_ =open('building.json','w')
    f_.write(jsondata2)
    f_.close() 

    jsondata3= json.dumps(dict3,indent=4,separators=(',',':'))
    f_ = open('highway.json','w')
    f_.write(jsondata3)
    f_.close()   

    jsondata4= json.dumps(dict4,indent=4,separators=(',',':'))
    f_ = open('railway.json','w')
    f_.write(jsondata4)
    f_.close() 

def norms(data):
# 1024
    return (data[0]*4,data[1]*4)

def create_func(path):
    file_list =os.listdir(path)
    image_size=[512,512]
    with open('/data/func.json') as f1:
        gj1 =json.load(f1)
    
    labels = list(gj1.keys())
    print(labels)
    for file_ in file_list:
        name,_ = os.path.splitext(file_)
        file_name = os.path.join(path,file_)

        if name.split("_")[1] =='func':
            with open(file_name) as f:
                gj = json.load(f)
            img = Image.new("L",image_size,'black')
            imgs =ImageDraw.Draw(img)

            for i in range(len(gj['label'])):
                for j in range(len(gj['data'][i])):
                    xy_data = gj['data'][i][j]
                    xy_new = []
                    for m in range(len(xy_data)):
                        xy=norms(xy_data[m])
                        xy_new.append(xy)
                    
                    imgs.polygon(xy_new, fill =(labels.index(gj['label'][i])+1))
        
            img.save("/data/work2/visual/{}.png".format(name))
        else:
            print("others")


def create_building(path):
    file_list =os.listdir(path)
    image_size=[512,512]
    with open('/data/building.json') as f1:
        gj1 =json.load(f1)
    
    labels = list(gj1.keys())
    print(labels)
    for file_ in file_list:
        name,_ = os.path.splitext(file_)
        file_name = os.path.join(path,file_)

        if name.split("_")[1] =='building':
            with open(file_name) as f:
                gj = json.load(f)
            img = Image.new("L",image_size,'black')
            imgs =ImageDraw.Draw(img)

            for i in range(len(gj['label'])):
                for j in range(len(gj['data'][i])):
                    xy_data = gj['data'][i][j]
                    xy_new = []
                    for m in range(len(xy_data)):
                        xy=norms(xy_data[m])
                        xy_new.append(xy)
                    
                    imgs.polygon(xy_new, fill =(labels.index(gj['label'][i])+1))
            img2 =np.array(img)
            np.save("/data/work2/visual/{}.npy".format(name),img2)
            
        else:
            print("others")


def create_highway(path):
    file_list =os.listdir(path)
    image_size=[512,512]
    with open('/data/highway.json') as f1:
        gj1 =json.load(f1)
    
    labels = list(gj1.keys())
    print(labels)
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        file_name = os.path.join(path,file_)
        if name.split('_')[1]=='highway':
            with open(file_name) as f :
                gj =json.load(f)
            
            img = Image.new("L",image_size,'black')
            imgs = ImageDraw.Draw(img)

            for i in range(len(gj['label'])):
                xy_data =gj['data'][i]
                xy_new =[]
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                
                imgs.line(xy_new, fill=(labels.index(gj['label'][i])+1))

            img.save("/data/work2/visual/{}.png".format(name))
        else:
            print("others")


def get_training_data(path1,path2):
    file_list =os.listdir(path1)
    file_list2 =os.listdir(path2)

    for file_ in file_list:
        name,_ = os.path.splitext(file_)
        func_ ="{}_func.png".format(name)
        func_file = os.path.join(path2,func_)
        highway_="{}_highway.png".format(name)
        high_file =os.path.join(path2,highway_)
        building_ ="{}_building.npy".format(name)
        build_file = os.path.join(path2,building_)

        func_i =np.array(Image.open(func_file))
        high_i =np.array(Image.open(high_file))
        build_i =np.load(build_file)

        new= np.stack((high_i,build_i,func_i),axis=0)

        print(new.shape)
        np.save("/data/work2/numpy_data/{}.npy".format(name),new)

def aug_data(path):
    file_list=os.listdir(path)
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        file_path =os.path.join(path,file_)
        new = np.load(file_path)
        for i in range(10):
            new_1 =new[:,32*(i):64+32*(i+1),32*(i):64+32*(i+1)]
            np.save("/data/work2/aug_numpy/{}_{}.npy".format(name,i),new_1)
 
 
def get_caption(path):
    file_list =os.listdir(path)
    with open('/data/highway.json') as f1:
        gj1 =json.load(f1)
    
    highways_l = list(gj1.keys())

    with open('/data/building.json') as f1:
        gj1 =json.load(f1)
    
    builing_l = list(gj1.keys())

    with open('/data/func.json') as f1:
        gj1 =json.load(f1)
    
    func_l = list(gj1.keys())

    for file_ in file_list:
        dict={'func':[],'highway':[],'building':[]}
        name,_ =os.path.splitext(file_)
        file_path = os.path.join(path,file_)
        new= np.load(file_path)
        h =new[0]
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if h[i,j]!=0:
                    label =highways_l[h[i,j]-1]
                    if label in dict['highway']:
                        print('exit')
                    else:
                        dict['highway'].append(label)
        b=new[1]

        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                if b[i,j]!=0:
                    label =builing_l[b[i,j]-1]
                    if label in dict['building']:
                        print('exit')
                    else:
                        dict['building'].append(label)
        
        f=new[2]
        dict['func'].append(func_l[f[32,32]-1])
        '''
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i,j]!=0:
                    label =func_l[f[i,j]-1]
                    if label in dict['func']:
                        print('exit')
                    else:
                        dict['func'].append(label)
        '''

        jsondata4= json.dumps(dict,indent=4,separators=(',',':'))
        f_ = open('/data/work2/caption/{}.json'.format(name),'w')
        f_.write(jsondata4)
        f_.close()     


def pure_func(path):
    file_list=os.listdir(path)
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        file_name =os.path.join(path,file_)

        old=np.load(file_name)
        h=old[0]
        b=old[1]
        f =np.ones([old[2].shape[0],old[2].shape[1]])*old[2][32,32]
        new=np.stack((h,b,f),axis=0)

        np.save('/data/work2/pure_func/{}.npy'.format(name),new)

def test(path):
    a=np.load(path)
    print(a)

def create_caption(path):
    file_list=os.listdir(path)
    dict={}
    dict2={}
    dict3={}
    for file_ in file_list:
        name,_ = os.path.splitext(file_)
        file_name =os.path.join(path,file_)
        with open(file_name) as f:
            gj=json.load(f)
        
        new_file_name="{}.npy".format(name)
        caption = 'a map of a {} region with '.format(gj['func'][0])
        caption1 = 'a map of a {} region with '.format(gj['func'][0])
        caption2=""
        caption3=''
        for i in range(len(gj['highway'])):
            if i ==0:
                caption =caption+"{} road ".format(gj['highway'][0])
                caption2 =caption1 +"{} road ".format(gj['highway'][0])
            else:
                caption =caption +"and {} road ".format(gj['highway'][i])
                caption2 =caption2+"and {} road ".format(gj['highway'][i])
        #for h in gj['highway']:
        #    caption =caption+"{} road ".format(h)
        #for j in range
        for b in range(len(gj['building'])):
            caption =caption + "and {} building footprint ".format(gj['building'][b])
            if b ==0:
                caption3 =caption1+ "{} building footprint ".format(gj['building'][b])
            else:
                caption3 =caption3+ "and {} building footprint ".format(gj['building'][b])
            
            #caption =caption+"and "
            #caption =caption+"{} building footprint".format(b)
        
        dict[new_file_name]=caption
        dict2[new_file_name]=caption2
        dict3[new_file_name]=caption3
    jsondata4= json.dumps(dict,indent=4,separators=(',',':'))
    f_ = open('/data/new_data/caption/caption.json'.format(name),'w')
    f_.write(jsondata4)
    f_.close() 

    jsondata4= json.dumps(dict2,indent=4,separators=(',',':'))
    f_ = open('/data/new_data/caption/caption_r.json'.format(name),'w')
    f_.write(jsondata4)
    f_.close() 

    jsondata4= json.dumps(dict3,indent=4,separators=(',',':'))
    f_ = open('/data/new_data/caption/caption_b.json'.format(name),'w')
    f_.write(jsondata4)
    f_.close() 

    

def refine_caption(path):
    with open("/data/new_data/caption/caption.json") as f:
        gj=json.load(f)
    with open("/data/new_data/caption/caption_r.json") as f2:
        gj2=json.load(f2)
    with open("/data/new_data/caption/caption_b.json") as f3:
        gj3 =json.load(f3)
    
    dict={}
    dict2={}
    dict3={}
    labels =list(gj.keys())
    for l in labels:
        cap =gj[l]
        cap2=gj2[l]
        cap3 =gj3[l]
        t=1
        for i in cap:
            if i in " ":
                t=t+1
        if t <60:
            dict[l]=cap
            dict2[l]=cap2
            dict3[l]=cap3
    jsondata4= json.dumps(dict,indent=4,separators=(',',':'))
    f_ = open('/data/new_data/caption/pure_caption.json','w')
    f_.write(jsondata4)
    f_.close() 
    jsondata4= json.dumps(dict2,indent=4,separators=(',',':'))
    f_ = open('/data/new_data/caption/pure_caption_r.json','w')
    f_.write(jsondata4)
    f_.close() 
    jsondata4= json.dumps(dict3,indent=4,separators=(',',':'))
    f_ = open('/data/new_data/caption/pure_caption_b.json','w')
    f_.write(jsondata4)
    f_.close() 

def ceate_image(path):
    file_list=os.listdir(path)
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        file_name =os.path.join(path,file_)
        new =np.load(file_name)
        h = new[0]
        b = new[1]
        f = new[2]
        img1 =Image.fromarray(h).convert('L')
        img2 =Image.fromarray(b).convert('L')
        img3 = Image.fromarray(f).convert('L')
        img1.save('/data/work2/visual_aug/{}_highway.png'.format(name))
        img2.save('/data/work2/visual_aug/{}_building.png'.format(name))
        img3.save('/data/work2/visual_aug/{}_func.png'.format(name))

def visual_highdata(path):
    file_list =os.listdir(path)
    image_size=[2048,2048]
    with open('/data/highway.json') as f1:
        gj1 =json.load(f1)
    
    labels = list(gj1.keys())
    line_width = [3,1,1,1,2,2,1,2,1,1,1,1,1,2,3,2,1,1,1,1,2,1,1,1,3,3,1,1,1]
    line_color = ['#111111','#7c7a80','#ff7f00','#ff7f00']
    print(labels)
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        file_name = os.path.join(path,file_)
        file_name2 = os.path.join("/data/work2/highway",file_)
        if name.split('_')[1]=='highway':
            #name
            new_name = name.split('_')[0]
            with open(file_name) as f :
                gj =json.load(f)
            with open(file_name2) as f:
                gj2 =json.load(f)
            img = Image.new("RGB",image_size,"#b3cb7a")
            imgs = ImageDraw.Draw(img)

            for i in range(len(gj['label'])):
                xy_data =gj['data'][i]
                xy_new =[]
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                #imgs.line(xy_new, width=line_width[labels.index(gj['label'][i])],fill='white')
                #imgs.line(xy_new, fill=(labels.index(gj['label'][i])+1))
                imgs.line(xy_new, width=line_width[labels.index(gj['label'][i])]*4,fill=line_color[line_width[labels.index(gj['label'][i])]]) 
            
            for i in range(len(gj2['label'])):
                xy_data =gj2['data'][i]
                xy_new =[]
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                #imgs.line(xy_new, width=line_width[labels.index(gj['label'][i])],fill='white')
                #imgs.line(xy_new, fill=(labels.index(gj['label'][i])+1))
                imgs.line(xy_new, width=line_width[labels.index(gj2['label'][i])]*4,fill=line_color[line_width[labels.index(gj2['label'][i])]]) 
            
            
            img.save("/data/work2/visual_highway3/{}.png".format(new_name))
        else:
            print("others")

def visual_building(path):
    file_list=os.listdir(path)
    image_size =[2048,2048]
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        building_path ="/data/work2/trans_data/{}_building.json".format(name)
        with open(building_path) as f2:
            gj2 =json.load(f2)
        img =Image.new("RGB",image_size,"#b3cb7a")
        imgs =ImageDraw.Draw(img)
        for i in range(len(gj2['label'])):
            for j in range(len(gj2['data'][i])):
                xy_data = gj2['data'][i][j]
                xy_new = []
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                
                imgs.polygon(xy_new, fill ='white')
        #img.save("/data/work2/visual_building_wo/{}_building.png".format(name))
        img.save("/data/work2/visual_building3/{}.png".format(name))


def visual_data(path):
    file_list=os.listdir(path)
    image_size =[2048,2048]
    with open('/data/highway.json') as f1:
        gj1 =json.load(f1)

    labels = list(gj1.keys())
    line_width = [3,1,1,1,2,2,1,2,1,1,1,1,1,2,3,2,1,1,1,1,2,1,1,1,3,3,1,1,1]
    line_color = ['#111111','#7c7a80','#ff7f00','#ff7f00']
    print(labels)  
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        #qianzhui=name.split('_')[0]
        print(name)
        highway_path = "/data/work2/trans_data/{}_highway.json".format(name)
        building_path ="/data/work2/trans_data/{}_building.json".format(name)
        extract_path = "/data/work2/highway/{}_highway.json".format(name)
        with open(highway_path) as f :
            gj =json.load(f)
        
        with open(building_path) as f2:
            gj2 =json.load(f2)
        with open(extract_path) as f3:
            gj3=json.load(f3)
        img =Image.new("RGB",image_size,"#b3cb7a")
        imgs =ImageDraw.Draw(img)
        for i in range(len(gj2['label'])):
            for j in range(len(gj2['data'][i])):
                xy_data = gj2['data'][i][j]
                xy_new = []
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                
                imgs.polygon(xy_new, fill ='white')

        for i in range(len(gj['label'])):
                xy_data =gj['data'][i]
                xy_new =[]
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                imgs.line(xy_new, width=line_width[labels.index(gj['label'][i])]*4,fill=line_color[line_width[labels.index(gj['label'][i])]])
        
        for i in range(len(gj3['label'])):
                xy_data =gj3['data'][i]
                xy_new =[]
                for m in range(len(xy_data)):
                    xy=norms(xy_data[m])
                    xy_new.append(xy)
                imgs.line(xy_new, width=line_width[labels.index(gj3['label'][i])]*4,fill=line_color[line_width[labels.index(gj3['label'][i])]])
        



         
        img.save("/data/work2/visual_color2/{}.png".format(name))
        

def aug_color(path):
    file_list=os.listdir(path)
    for file_ in file_list:
        name,_ =os.path.splitext(file_)
        file_path =os.path.join(path,file_)
        new =Image.open(file_path).convert("RGB")
        
        new_np = np.array(new)
        print(new_np.shape)
        for i in range(10):
            #new_l = new_np[64*(i):128+64*(i+1),64*(i):128+64*(i+1),:]
            #蠢了
            new_l = new_np[128*(i):256+128*(i+1),128*(i):256+128*(i+1),:]
            im = Image.fromarray(np.uint8(new_l))
            im.save('/data/work2/aug_color2/{}_{}.png'.format(name.split('_')[0],i))   

def create_mask(path):
    file_list=os.listdir(path)
    for file_ in file_list:
        name,_ = os.path.splitext(file_)
        img =Image.new("RGB", [384,384],'black')
        imgs =ImageDraw.Draw(img)
        imgs.polygon([(128,128),(256,128),(256,256),(128,256)],fill='white')
        img.save('/data/work2/mask/{}.png'.format(name))

def extract_highway(path):
    file_list = os.listdir(path)

    for file_ in file_list:
        name, _ =os.path.splitext(file_)
        file_name = os.path.join(path,file_)
        
        if name.split('_')[1]=='highway':
            json1 ={"coor_base":[
            0,
            512,
            512,
            0
            ],"label":[],"data":[]}
            with open(file_name) as f :
                gj =json.load(f)
            for i in range(len(gj["label"])):
                if gj["label"][i] =="trunk":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="tertiary":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="primary":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="secondary":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="primary_link":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="trunk_link":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="secondary_link":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="tertiary_link":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="motorway":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                elif gj["label"][i] =="motorway_link":
                    json1["label"].append(gj["label"][i])
                    json1["data"].append(gj["data"][i])
                else:
                    print("1")
            jsondata4= json.dumps(json1,indent=4,separators=(',',':'))
            f_ = open('/data/work2/highway/{}.json'.format(name),'w')
            f_.write(jsondata4)
            f_.close() 

import random
def train_valid(path):
    file_names = os.listdir(path)
    random.shuffle(file_names)
    length_ = len(file_names)
    train_l = int(length_ *0.9)
    train_names = file_names[:train_l]
    test_names = file_names[train_l:]

    train_file = open("/data/work2/train.txt","w")
    for line in train_names:
        train_file.write(line+"\n")
    #train_file.writelines(train_names)
    train_file.close()
    test_file = open("/data/work2/valid.txt","w")
    for line in test_names:
        test_file.write(line+"\n")

    #test_file.writelines(test_names)
    test_file.close()


def large_img(path,new_path):
    file_names =os.listdir(path)
    print(file_names)
    random.seed(42)
    for file_ in file_names:
        name,_ = os.path.splitext(file_)
        file_name = os.path.join(path, file_)
        img = Image.open(file_name)
        for i in range(50):
            x = random.randint(0,1022)
            y = random.randint(0,1022)
            new_img = img.crop((x,y,x+1024,y+1024))
            new_names = os.path.join(new_path,'{}_{}.png'.format(name,i))
            new_img.save(new_names)

import shutil

def copy_file(old_path, new_path):
    file_names = os.listdir('/sample/mask')

    for file_ in file_names:
        file_name = os.path.join(old_path, file_)
        new_file_name = os.path.join(new_path, file_)
        shutil.copyfile(file_name, new_file_name)

#512-->256
def convert_5_2(old_path, new_path):
    file_names = os.listdir(old_path)

    for file_ in file_names:
        file_name =os.path.join(old_path,file_)
        img = Image.open(file_name).convert('RGB')
        new_img = img.resize([400,400],resample=Image.NEAREST)
        new_img = img.resize([256,256],Image.NEAREST)
        #new_img = img.resize([256,256],Image.LANCZOS)
        new_file_name = os.path.join(new_path,file_)
        new_img.save(new_file_name)


        

if __name__ =='__main__':

    #visual_highdata('/data/work2/trans_data')
    #visual_building('/data/work2/raw_data')
    #visual_data('/data/work2/raw_data')
    #aug_color('/data/work2/visual_color2')
    #create_mask('/data/work2/aug_building')
    #extract_highway('/data/work2/trans_data')

    #train_valid("/data/work2/aug_color2")

    #large image
    #large_img('/data/work2/visual_building3','/data/work2/large_b')

    #copy file
    #copy_file('/data/work2/large_b','/sample/b_gt')
    #copy_file('/data/work2/large_h','/sample/road_gt')

    #resize
    convert_5_2('/sample/road_gt','/sample_256/road_gt')