
import pickle
def load_pickle(path):
    with open(path,'rb') as file:
        data=pickle.load(file)
    return data

def write_pickle(data,path):
    with open(path,'wb') as file:
        pickle.dump(data, file)

def append_record(bedpe_path_min,min_loc_list,chrom,resolution=10000):
    if "_" in chrom:
        chrom = chrom.split("_")[0]
    
    with open(bedpe_path_min,'a') as wfile:
        for loc in min_loc_list:
            x1,x2 = loc
            if x1<x2:
                #skip any diagonal detections
                wfile.write("%s\t%d\t%d\t%s\t%d\t%d\n"%(chrom,x1*resolution, (x1+1)*resolution,
                                chrom,x2*resolution,(x2+1)*resolution))