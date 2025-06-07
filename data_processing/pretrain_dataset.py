import os
import numpy as np
import torch
import torch.utils.data
import random
from collections import defaultdict
from scipy.sparse import coo_matrix
import pickle 
from ops.sparse_ops import array_to_coo
from ops.io_utils import load_pickle
from data_processing.finetune_dataset import to_tensor, list_to_tensor

def validate_input_size(input_matrix, window_height, window_width):
    """
    Validate the input size is larger than the window size
    Args:
        input_matrix: the input matrix
        window_height: the height of the window
        window_width: the width of the window
    """
    if isinstance(input_matrix, coo_matrix):
        input_matrix = input_matrix.toarray()
    input_height, input_width = input_matrix.shape
    if input_height>=window_height and input_width>=window_width:
        #this validation is different from fine-tuning since we can crop the input to self-supervise
        return True
    return False 

def sample_index(matrix_size,window_size):
    """
    Sample the index of the window
    Args:
        matrix_size: the size of the matrix
        window_size: the size of the window
    """
    if matrix_size==window_size:
        return 0
    start = random.randint(0, matrix_size-window_size-1)
    return start

def sample_index_patch(matrix_size,window_size,patch_size):
    """
    Please choose this version if you want to use hi-c processed data pipeline
    The generated patch make sure the diagonal region only starts at the multiple of patch_size
    Sample the index of the window only in patch separation
    Args:
        matrix_size: the size of the matrix
        window_size: the size of the window
        patch_size: the size of the patch
    """
    if matrix_size==window_size:
        return 0
    patch_list=[]
    for i in range(0,matrix_size-window_size,patch_size):
        patch_list.append(i)
    start = random.choice(patch_list)
    return start

def sample_diag_index(matrix_size,window_size,diagonal_index,patch_size):
    """
    Sample the index of the window
    Args:
        matrix_size: the size of the matrix
        window_size: the size of the window
        diagonal_index: the index of the diagonal
    """
    if diagonal_index<0 or diagonal_index>=matrix_size:
        #invalid diagonal index
        return sample_index(matrix_size,window_size)
    candidate_list=[diagonal_index]
    left_index= diagonal_index
    while left_index>0:
        left_index = left_index - patch_size
        if left_index>=0:
            candidate_list.append(left_index)
        else:
            break
    right_index = diagonal_index
    while right_index<matrix_size:
        right_index = right_index + patch_size
        if right_index<matrix_size:
            candidate_list.append(right_index)
        else:
            break
    candidate_list = sorted(candidate_list)
    final_candidate_list= [index for index in candidate_list if index+window_size<=matrix_size]
    if len(final_candidate_list)==0:
        final_candidate_list = candidate_list[0:1]
    start = random.choice(final_candidate_list)
    return start
class Pretrain_Dataset(torch.utils.data.Dataset):
    def __init__(self,data_list,   
                transform=None,
                sparsity_filter=0.05,
                patch_size=16,
                window_height= 224,
                window_width = 224):
        """
        Args:
            data_list: the list of data
            transform: the transformation function
            sparsity_filter: the sparsity ratio to filter too sparse data for pre-training
            window_height: the height of the window
            window_width: the width of the window
        """
        
        self.data_list = data_list
        self.transform = transform
        self.window_height = window_height
        self.window_width = window_width
        self.sparsity_filter = sparsity_filter
        self.patch_size = patch_size
        # self.input_count_flag = False
        self.train_dict=defaultdict(list)
        self.train_list=[]
        for data_index, data_dir in enumerate(data_list):
            cur_dir = data_dir
            dataset_name = os.path.basename(cur_dir)
            listfiles = os.listdir(cur_dir)
            listfiles = sorted(listfiles)
            for file_index,file in enumerate(listfiles):
                cur_path = os.path.join(cur_dir, file)
                if file.endswith('.pkl'):
                    if file_index==0:
                        #verify the input pkl file includes the input key
                        data= load_pickle(cur_path)
                        data_keys = list(data.keys())
                        if 'input' not in data:
                            print("The input key is not included in the pkl file. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            continue
                        #check input_count key
                        # if 'input_count' in data:
                        #     self.input_count_flag = True
                        # if 'input_count' not in data:
                        #     print("The input_count key is not included in the pkl file. The directory is skipped.")
                        #     print("The dir is {}".format(cur_dir))
                        #     continue
                        
                        #validate the input size
                        input_matrix = data['input']
                        if not validate_input_size(input_matrix, window_height, window_width):
                            print("The input size is not matched with the window size. The directory is skipped.")
                            print("The dir is {}".format(cur_dir))
                            print("The input size is {}".format(input_matrix.shape))
                            print("The specified window size is {} x {}".format(window_height, window_width))
                            print("Please adjust --input_row_size and --input_col_size to match your input.")
                            continue
                    self.train_dict[dataset_name].append(cur_path)
                    self.train_list.append(cur_path)
                else:
                    print("The file {} is not a .pkl file.".format(file),"It is skipped.")
                    continue    
        print("The number of samples used in the dataset is {}".format(len(self.train_list)))
        #print("Use count flag is {}".format(self.input_count_flag))
    #you can either select the train_list or train_dict to do training based on your exprience
    def __len__(self):
        return len(self.train_list)
    
    def convert_rgb(self,data_log,max_value):
        if len(data_log.shape)==2:
            data_log = data_log[np.newaxis,:]
        data_red = np.ones(data_log.shape)
        data_log1 = (max_value-data_log)/max_value
        data_rgb = np.concatenate([data_red,data_log1,data_log1],axis=0,dtype=np.float32)#transform only accept channel last case
        data_rgb = data_rgb.transpose(1,2,0)
        return data_rgb
    
    def __getitem__(self, idx):
        """
        Args:
            idx: the index of the data
        """
        data_path = self.train_list[idx]
        data= load_pickle(data_path)
        input_matrix = data['input']
        region_size = input_matrix.shape[0]*input_matrix.shape[1]
        if isinstance(input_matrix, coo_matrix):
            cur_sparsity = input_matrix.nnz/region_size
        else:
            cur_sparsity = np.count_nonzero(input_matrix)/region_size
        #we suggest you processed the submatrix to make sure they pass the threshold, otherwise it may take much longer to iteratively sampling until passing the threshold
        if cur_sparsity<self.sparsity_filter:
            random_index = random.randint(0, len(self.train_list)-1)
            return self.__getitem__(random_index)
        
        if isinstance(input_matrix, coo_matrix):
            input_matrix = input_matrix.toarray()
            #make sure you save the down-diagonal regions if you use the coo_matrix
            #to support off-diagonal submatrix, we did not any automatic symmetrical conversion for your input array.

        input_matrix = np.nan_to_num(input_matrix)
        
        if 'input_count' in data:
            matrix_count = np.sum(input_matrix)
            hic_count = data['input_count']
            if hic_count is None:
                hic_count = 1000000000
        
        else:
            hic_count = 1000000000 #as placeholder for cases user some data without input_count but some with
            matrix_count = np.sum(input_matrix)
        hic_count = float(hic_count)

        submat = np.zeros([1,self.window_height,self.window_width])

        #judge if we need to use diag or not
        use_diag_flag=False

        if "diag" not in data:        
            pass
        else:
            diag = data['diag']
            if diag is None:
                pass
            elif (diag<0 and abs(diag)>=input_matrix.shape[0]):
                pass
            elif (diag>0 and diag>=input_matrix.shape[1]):
                pass
            else:
                use_diag_flag=True
        
        if not use_diag_flag:
            row_start = sample_index(input_matrix.shape[0],self.window_height)
            col_start = sample_index(input_matrix.shape[1],self.window_width)
            return_diag= max(self.window_height,self.window_width)+1#indicating no diag needed to use
        else:
            M,N = input_matrix.shape
            if diag<0:
                #diag starts at [diag,0], ends [M-1,M-1-abs(diag)] if M<N
                #else have the possibility diag starts at [diag,0], ends [N-1+diag,N-1] if M>=N
                diag = abs(diag) #make sure diag is positive to do calculation
                if M-abs(diag)<N:
                    #first sample col_start, to check if diagonal region is included
                    col_start = sample_index(input_matrix.shape[1],self.window_width)
                    diag_col_end = input_matrix.shape[0]-abs(diag)
                    if col_start>=diag_col_end:
                        row_start = sample_index(input_matrix.shape[0],self.window_height)
                        return_diag= max(self.window_height,self.window_width)+1 #no diag needed to use
                    else:
                        #diagonal region is included
                        #use the diagonal index and row start to define the row_start we can use
                        #make sure the diagonal region is in the patch boundary
                        search_diag = abs(diag)+col_start
                        row_start=sample_diag_index(input_matrix.shape[0],self.window_height,search_diag,self.patch_size)
                        return_diag = row_start-search_diag
                        #if it is positive, then boundary starts in row of submatrix, if it is negative, then boundary starts in col of submatrix
                        #here we limits the freedom to not consider the bigger randomness if abs(diag)>self.window_height, then the row_start can be have more choices
                        #because col_start is already full degree of randomness
                else:
                    #diag starts at [diag,0], ends [diag+N-1,N-1] if M>=N
                    #consider only possiblility of diagonal region is not included
                    row_start = sample_index(input_matrix.shape[0],self.window_height)
                    if row_start+self.window_height<abs(diag) or row_start>=M-abs(diag)-N:
                        col_start = sample_index(input_matrix.shape[1],self.window_width)
                        return_diag= max(self.window_height,self.window_width)+1
                    else:
                        #diagonal region is included
                        search_diag = row_start-abs(diag)
                        if search_diag>0:
                            col_start=sample_diag_index(input_matrix.shape[1],self.window_width,search_diag,self.patch_size)
                            return_diag = search_diag-col_start
                            #if it is positive, then boundary starts in row of submatrix, if it is negative, then boundary starts in col of submatrix
                        else:
                            #full random of col_start, then control row_start to make sure diagonal in the patch boundary
                            col_start = sample_index(input_matrix.shape[1],self.window_width)
                            search_diag=col_start+abs(diag)
                            row_start=sample_diag_index(input_matrix.shape[0],self.window_height,search_diag,self.patch_size)
                            return_diag = row_start-search_diag
            else:
                #diag starts at [0,diag], ends [M-1,M-1+diag] if M<N
                #else have the possiblity diag starts at [0,diag], ends [N-1-diag,N-1] if M>=N
                diag=abs(diag)
                if M+diag<N:
                    #1st situation: diag starts at [0,diag], ends [M-1,M-1+diag]
                    col_start = sample_index(input_matrix.shape[1],self.window_width)
                    diag_col_left = diag
                    diag_col_right= M+diag
                    if col_start+self.window_width<=diag_col_left or col_start>=diag_col_right:
                        row_start = sample_index(input_matrix.shape[0],self.window_height)
                        return_diag= max(self.window_height,self.window_width)+1
                    else:
                        #diagonal region is included
                        search_diag = col_start-diag
                        if search_diag>=0:
                            row_start=sample_diag_index(input_matrix.shape[0],self.window_height,search_diag,self.patch_size)
                            return_diag=row_start-search_diag
                        else:
                            #full random of row_start, then control col_start to make sure diagonal in the patch boundary
                            row_start = sample_index(input_matrix.shape[0],self.window_height)
                            search_diag=row_start+diag
                            col_start=sample_diag_index(input_matrix.shape[1],self.window_width,search_diag,self.patch_size)
                            return_diag = search_diag-col_start
                else:
                    #2nd condition: diag starts at [0,diag], ends [N-1-diag,N-1]
                    row_start = sample_index(input_matrix.shape[0],self.window_height)
                    diag_row_end = N-diag
                    if row_start>=diag_row_end:
                        col_start = sample_index(input_matrix.shape[1],self.window_width)
                        return_diag= max(self.window_height,self.window_width)+1
                    else:
                        #diagonal region is included
                        search_diag = row_start+diag
                        col_start=sample_diag_index(input_matrix.shape[1],self.window_width,search_diag,self.patch_size)
                        return_diag = search_diag-col_start
                        #here we limits the freedom to not consider the bigger randomness if abs(diag)>self.window_width, then the col_start can be have more choices
                        #because row_start is already full degree of randomness
        row_end = min(row_start+self.window_height,input_matrix.shape[0])
        col_end = min(col_start+self.window_width,input_matrix.shape[1])
        submat[0,0:row_end-row_start,0:col_end-col_start] = input_matrix[row_start:row_end,col_start:col_end] 
        submat = submat.astype(np.float32)
        mask_array = np.ones(submat.shape,dtype=np.float32)
        mask_array[submat==0]=0
        mask_array = mask_array[np.newaxis,:,:]
        cur_sparsity = np.sum(mask_array)/self.window_height/self.window_width
        input = submat
        max_value = np.max(input)
        if cur_sparsity<self.sparsity_filter or max_value<0.01:
            random_index = random.randint(0, len(self.train_list)-1)
            return self.__getitem__(random_index)
        
        input = np.log10(input+1)
        max_value = np.log10(max_value+1)
        input = self.convert_rgb(input,max_value)
        if self.transform is not None:
            input = self.transform(input)
        # convert the diagonal position from pixel unit into patch unit
        return_diag = int(return_diag / self.patch_size)
        
        return list_to_tensor([input, mask_array, hic_count, return_diag,matrix_count])