import numpy as np
def get_mask(a,start=None, end= None, index= None, repeat = None):
    empty_mask= np.zeros_like(a)
    print('empty mask ',empty_mask.shape)
    if index is not None: 
        print('index ',index)
        empty_mask[:,index]=1
        print('empty mask 2 ',empty_mask.shape)
        #if repeat is not None: 
        #    empty_mask=np.repeat(empty_mask, repeat)
    else: 
        if repeat is None: 
            #i=0 
            for i in range(0, len(empty_mask)):
                empty_mask[i,int(start[i][0]):int(start[i][1]),int(end[i][0]):int(end[i][1])]=1 
            #i+=1
        else: 
            for a in start: 
                for  b in end: 
                    empty_mask[a:b]=1 
            empty_mask=np.repeat(empty_mask,repeat)
    print('empty mask END ',empty_mask.shape)
    return empty_mask