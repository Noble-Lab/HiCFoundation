

import numpy as np
from scipy.sparse import coo_matrix
import hicstraw
import os
import pickle 
def write_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
def read_chrom_array(chr1, chr2, normalization, hic_file, resolution,call_resolution):
    chr1_name = chr1.name
    chr2_name = chr2.name
    infos = []
    infos.append('observed')
    infos.append(normalization)
    infos.append(hic_file)
    infos.append(chr1_name)
    infos.append(chr2_name)
    infos.append('BP')
    infos.append(call_resolution)
    print(infos)
    row, col, val = [], [], []
    rets = hicstraw.straw(*infos)
    print('\tlen(rets): {:3e}'.format(len(rets)))
    for ret in rets:
        row.append((int)(ret.binX // resolution))
        col.append((int)(ret.binY // resolution))
        val.append(ret.counts)
    print('\tsum(val): {:3e}'.format(sum(val)))
    if sum(val) == 0:
        return None
    if chr1_name==chr2_name:
        max_shape =max(max(row),max(col))+1
        mat_coo = coo_matrix((val, (row, col)), shape = (max_shape,max_shape),dtype=np.float32)
    else:
        max_row = max(row)+1
        max_column = max(col)+1
        mat_coo = coo_matrix((val, (row, col)), shape = (max_row,max_column),dtype=np.float32)

    mat_coo = mat_coo #+ triu(mat_coo, 1).T #no below diagonaline records

    return mat_coo

def validate_resolution(resolution,resolution_list):
    for reso in resolution_list:
        if resolution%reso==0:
            return True
    return False

def find_call_resolution(resolution, resolution_list):
    max_best_resolution = min(resolution_list)
    resolution_list.sort()
    for reso in resolution_list:
        if resolution%reso==0:
            if reso>max_best_resolution:
                max_best_resolution=reso
    return max_best_resolution

def hic2array(input_hic,output_pkl=None,
              resolution=25000,normalization="NONE",
              tondarray=0):
    """
    input_hic: str, input hic file path
    output_pkl: str, output pickle file path
    resolution: int, resolution of the hic file
    """

    hic = hicstraw.HiCFile(input_hic)
    chrom_list=[]
    chrom_dict={}
    for chrom in hic.getChromosomes():
        print(chrom.name, chrom.length)
        if "all" in chrom.name.lower():
            continue
        chrom_list.append(chrom)
        chrom_dict[chrom.name]=chrom.length
    resolution_list = hic.getResolutions()
    max_resolution_candidate = max(resolution_list)
    if not validate_resolution(resolution):
        print("Resolution not found in the hic file, please choose from the following list:")
        print(resolution_list)
        print("Any other resolution is coarse than this and dividable by one of them can also be supported.")
        exit()
    output_dict={}
    for i in range(len(chrom_list)):
        for j in range(i,len(chrom_list)):
            if i!=j and tondarray in [2,3]:
                #skip inter-chromosome region
                continue
            
            chrom1 = chrom_list[i]
            chrom1_name = chrom_list[i].name
            chrom2 = chrom_list[j]
            chrom2_name = chrom_list[j].name
            if 'Un' in chrom1_name or 'Un' in chrom2_name:
                continue
            if "random" in chrom1_name.lower() or "random" in chrom2_name.lower():
                continue
            if "alt" in chrom1_name.lower() or "alt" in chrom2_name.lower():
                continue
            read_array=read_chrom_array(chrom1,chrom2, normalization, input_hic, resolution,call_resolution=find_call_resolution(resolution,resolution_list))
            if read_array is None:
                print("No data found for",chrom1_name,chrom2_name)
                continue
            if tondarray in [1,3]:
                read_array = read_array.toarray()
            if tondarray in [2,3]:
                output_dict[chrom1_name]=read_array
            else:
                output_dict[chrom1_name+"_"+chrom2_name]=read_array
    if output_pkl is not None:
        output_dir = os.path.dirname(os.path.realpath(output_pkl))
        os.makedirs(output_dir, exist_ok=True)
        write_pkl(output_dict,output_pkl)

    return output_dict
"""

Usage
```
python3 hic2array_simple.py [input.hic] [output.pkl] [resolution] [normalization_type] [mode]
```

This is the full cool2array script, converting both intra, inter chromosome regions to array format. <br>
The output array is saved in a pickle file as dict: [chrom1_chrom2]:[array] format. <br>
[resolution] is used to specify the resolution that stored in the output array. <br>
[normalization_type] supports the following type: <br>
```
0: NONE normalization applied, save the raw data to array.
1: VC normalization; 
2: VC_SQRT normalization; 
3: KR normalization; 
4: SCALE normalization.
```
Four modes are supported for different format saving: 
```
0: scipy coo_array format output; 
1: numpy array format output;
2: scipy csr_array format output (only include intra-chromsome region).
3: numpy array format output (only include intra-chromsome region).
```

"""
if __name__ == '__main__':
    import os 
    import sys
    if len(sys.argv) != 6:
        print('Usage: python3 hic2array_simple.py [input.hic] [output.pkl] [resolution] [normalization_type] [mode]')
        print("This is the full hic2array script. ")
        print("normalization type: 0: None normalization; 1: VC normalization; 2: VC_SQRT normalization; 3: KR normalization; 4: SCALE normalization")
        print("mode: 0 for sparse matrix, 1 for dense matrix, 2 for sparce matrix (only cis-contact); 3 for dense matrix (only cis-contact).")
        sys.exit(1)
    resolution = int(sys.argv[3])
    normalization_type = int(sys.argv[4])
    mode = int(sys.argv[5])
    normalization_dict={0:"NONE",1:"VC",2:"VC_SQRT",3:"KR",4:"SCALE"}
    if normalization_type not in normalization_dict:
        print('normalization type should be 0,1,2,3,4')
        print("normalization type: 0: None normalization; 1: VC normalization; 2: VC_SQRT normalization; 3: KR normalization; 4: SCALE normalization")
        sys.exit(1)
    normalization_type = normalization_dict[normalization_type]
    if mode not in [0,1,2,3]:
        print('mode should be in choice of 0/1/2/3')
        print("mode: 0 for sparse matrix, 1 for dense matrix, 2 for sparce matrix (only cis-contact); 3 for dense matrix (only cis-contact).")
        sys.exit(1)
    input_hic_path = os.path.abspath(sys.argv[1])
    output_pkl_path = os.path.abspath(sys.argv[2])
    output_dir = os.path.dirname(output_pkl_path)
    os.makedirs(output_dir,exist_ok=True)
    hic2array(input_hic_path,output_pkl_path,resolution,normalization_type,mode)
    