import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.nn as nn
import time
from ops.Logger import MetricLogger,SmoothedValue
import os
from collections import defaultdict
from ops.sparse_ops import array_to_coo
from scipy.sparse import coo_matrix,triu
def inference_worker(model,data_loader,log_dir=None,args=None):
    """
    model: model for inference
    data_loader: data loader for inference
    log_dir: log directory for inference
    args: arguments for inference
    """
    model.eval()
    config_resolution = args.resolution
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Inference: '
    print_freq = args.print_freq
    print("number of iterations: ",len(data_loader))
    num_iter = len(data_loader)
    dataset_shape_dict = data_loader.dataset.dataset_shape
    infer_task = args.task
    if infer_task==1:
        output_dict=defaultdict(list)
    elif infer_task==2 or infer_task==3 or infer_task==5:
        output_dict={}
        for chrom in dataset_shape_dict:
            output_dict[chrom] = {"row_record":[],"col_record":[],"value_record":[],"count_record":[]}
    elif infer_task==4:
        #epigenomic assay prediction
        num_track = 6
        output_dict={}
        for chrom in dataset_shape_dict:
            current_shape = dataset_shape_dict[chrom]
            current_length = current_shape[0]
            mean_array = np.zeros([num_track,current_length])
            count_array = np.zeros([num_track,current_length])
            output_dict[chrom] = {"mean":mean_array,"count":count_array}
    elif infer_task in [6, 7]:
        output_dict={"submat_embedding":defaultdict(list),
                     "patch_embedding":defaultdict(list),
                     "skipped_regions": []}

    if infer_task==3:
        #resolution enhancement
        cutoff= 1000
        cutoff = torch.tensor(cutoff).float().cuda()
        log_cutoff = torch.log10(cutoff+1).cuda()
    if infer_task==5:
        #scHi-C enhancement
        cutoff= 1000
        log_cutoff = np.log10(cutoff+1)
        output_dict={}
        for chrom in dataset_shape_dict:
            current_shape = dataset_shape_dict[chrom]
            current_length = current_shape[0]
            mean_array = np.zeros(current_shape)
            count_array = np.zeros(current_shape)
            output_dict[chrom] = {"mean":mean_array,"count":count_array}
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input,total_count,indexes = data
        input = input.cuda()
        input = input.float()
        total_count = total_count.cuda()
        total_count = total_count.float()
        with torch.no_grad():
            output = model(input,total_count) 
            # fixme: loop, and epigenomic assay prediction did not take count in benchmark, I think this will not impact performance, will check later. If yes, will revise it to model(input)
        if infer_task==1:
            #reproducibility analysis
            pass
        elif infer_task==2:
            #loop calling
            output= torch.sigmoid(output)
        elif infer_task==3:
            #resolution enhancement
            output = output*log_cutoff
            output = torch.pow(10,output)-1
            output = torch.clamp(output,min=0)

        elif infer_task in [6, 7]:
            #get the specified encoder/decoder layer's output
            output = output[args.embed_depth]


        # elif infer_task==5:
        #     #scHi-C enhancement
        #     output = output*log_cutoff
        #     output = torch.pow(10,output)-1
        #     output = torch.round(output)-2
        #     output = torch.clamp(output,min=0)

        output = output.detach().cpu().numpy()
        input = input.detach().cpu().numpy()
        chrs, row_starts, col_starts = indexes
        for i in range(len(output)):
            chr = chrs[i]
            row_start = row_starts[i]
            col_start = col_starts[i]
            row_start = int(row_start)
            col_start = int(col_start)
            row_start = max(0,row_start)
            col_start = max(0,col_start)
            current_shape = dataset_shape_dict[chr]
            row_end = min(row_start+args.input_row_size,current_shape[0])
            col_end = min(col_start+args.input_col_size,current_shape[1])
            current_input = input[i]
            #input_count = np.sum(current_input)
            #ignore empty matrix
            if np.isnan(np.sum(current_input)):
                print("empty matrix:",chr,row_start*config_resolution,row_end * config_resolution,col_start*config_resolution,col_end * config_resolution)
                if infer_task in [6, 7]:
                    output_dict["skipped_regions"].append([chr, 
                                                           row_start * config_resolution,
                                                           row_end * config_resolution,
                                                           chr,
                                                           col_start * config_resolution,
                                                           col_end * config_resolution])
                continue

            # # may be not necessary, will check if error happens
            # if input_count<=len(current_input):
            #     #skip super low read count matrix
            #     #that's to say, <1 read per 10 kb, samller than 0.3M total read for human
            #     continue
            cur_output = output[i]
            if infer_task==1:
                match_key = f"{chr}:{row_start*config_resolution},{col_start*config_resolution}"
                output_dict[match_key] = cur_output
            elif infer_task==2 or infer_task==3:
                #loop calling, resolution enhancement
                cur_output = cur_output[:row_end-row_start,:col_end-col_start]
                cur_output = array_to_coo(cur_output)
                output_dict[chr]["row_record"].append(cur_output.row+row_start)
                output_dict[chr]["col_record"].append(cur_output.col+col_start)
                output_dict[chr]["value_record"].append(cur_output.data)
                output_dict[chr]["count_record"].append([1]*len(cur_output.data))
            elif infer_task==4:
                #epigenomic assay prediction
                cur_output = cur_output[:, :row_end-row_start]
                output_dict[chr]['mean'][:, row_start:row_end] += cur_output
                output_dict[chr]['count'][:, row_start:row_end] += 1

            elif infer_task in [6, 7]:
                refer_row = row_start
                refer_col = col_start
                real_row_start = max(0,refer_row-args.input_row_size//2)
                real_col_start = max(0,refer_col-args.input_col_size//2)
                real_row_end = min(current_shape[0],refer_row+args.input_row_size//2)
                real_col_end = min(current_shape[1],refer_col+args.input_col_size//2)
                patch_row_range = (real_row_end-real_row_start)//args.patch_size
                patch_col_range = (real_col_end-real_col_start)//args.patch_size
                # cur_output = cur_output[:patch_row_range,:patch_col_range]
                # we can let the patch embedding choice.
                if args.patch_embedding:
                    for row_index in range(real_row_start,real_row_end, args.patch_size):
                        for col_index in range(real_col_start,real_col_end,args.patch_size):
                            row_index = int(row_index)
                            col_index = int(col_index)
                            patch_row_index = (row_index-real_row_start)//args.patch_size
                            patch_col_index = (col_index-real_col_start)//args.patch_size
                            cur_patch_embedding = cur_output[patch_row_index,patch_col_index]
                            middle_row = row_index+args.patch_size//2
                            middle_col = col_index+args.patch_size//2
                            search_key = f"{chr}:{middle_row*config_resolution},{middle_col*config_resolution}"
                            output_dict["patch_embedding"][search_key].append(cur_patch_embedding)
                            
                search_key = f"{chr}:{refer_row*config_resolution},{refer_col*config_resolution}"
                #average embedding
                all_embedding = cur_output.reshape(-1,cur_output.shape[-1])
                all_embedding = np.mean(all_embedding,axis=0)
                output_dict["submat_embedding"][search_key].append(all_embedding)


            elif infer_task == 5:
                #scHi-C enhancement
                if current_shape[0] < args.input_row_size or current_shape[1] < args.input_col_size:
                    #remove padding regions
                    left_up_pad_size = (args.input_row_size - current_shape[0]) // 2
                    right_down_pad_size = args.input_row_size - current_shape[0] - left_up_pad_size
                    left_up_pad_size_col = (args.input_col_size - current_shape[1]) // 2
                    right_down_pad_size_col = args.input_col_size - current_shape[1] - left_up_pad_size_col
                    cur_output = cur_output[left_up_pad_size:-right_down_pad_size, left_up_pad_size_col:-right_down_pad_size_col]
                    output_dict[chr]['mean'] += cur_output
                    output_dict[chr]['count'] += 1
                else:
                    output_dict[chr]['mean'][row_start:row_start+args.input_row_size, col_start:col_start+args.input_col_size] += cur_output
                    output_dict[chr]['count'][row_start:row_start+args.input_row_size, col_start:col_start+args.input_col_size] += 1
                # cur_output = array_to_coo(cur_output)
                # output_dict[chr]["row_record"].append(cur_output.row+row_start)
                # output_dict[chr]["col_record"].append(cur_output.col+col_start)
                # output_dict[chr]["value_record"].append(cur_output.data)
                # output_dict[chr]["count_record"].append([1]*len(cur_output.data))


    
    if infer_task==1:
        return output_dict
    elif infer_task==2 or infer_task==3:
        final_dict={}
        for chrom in output_dict:
            row_record = np.concatenate(output_dict[chrom]["row_record"])
            col_record = np.concatenate(output_dict[chrom]["col_record"])
            value_record = np.concatenate(output_dict[chrom]["value_record"])
            count_record = np.concatenate(output_dict[chrom]["count_record"])
            combine_row=np.concatenate([row_record,col_record])
            combine_col=np.concatenate([col_record,row_record])
            combine_value=np.concatenate([value_record,value_record])
            combine_count=np.concatenate([count_record,count_record])
            prediction_sym = coo_matrix((combine_value, (combine_row, combine_col)), shape=dataset_shape_dict[chrom])
            count_sym = coo_matrix((combine_count, (combine_row, combine_col)), shape=dataset_shape_dict[chrom])
            
            prediction_sym.sum_duplicates()
            count_sym.sum_duplicates()
            prediction_sym.data = prediction_sym.data/count_sym.data
            #remove very small prediction to save time
            select_index = prediction_sym.data>0.01
            prediction_sym.data = prediction_sym.data[select_index]
            prediction_sym.row = prediction_sym.row[select_index]
            prediction_sym.col = prediction_sym.col[select_index]
            print("finish summarize %s prediction"%chrom,prediction_sym.nnz)
            final_dict[chrom] = triu(prediction_sym,0)
        return final_dict
    elif infer_task==4:
        #epigenomic assay prediction
        return_dict={}
        for chrom in dataset_shape_dict:
            count_array=output_dict[chrom]['count']
            mean_array=output_dict[chrom]['mean']
            count_array =np.maximum(count_array,1)
            mean_array = mean_array/count_array
            mean_array = np.nan_to_num(mean_array)
            return_dict[chrom] = mean_array
        return return_dict
    elif infer_task == 5:
        return_dict={}
        for chrom in dataset_shape_dict:
            count_array=output_dict[chrom]['count']
            mean_array=output_dict[chrom]['mean']
            count_array =np.maximum(count_array,1)
            mean_array = (mean_array + mean_array.T)/2
            mean_array = mean_array/count_array
            mean_array = np.nan_to_num(mean_array)
            mean_array = mean_array*log_cutoff
            mean_array = np.power(10, mean_array) - 1
            mean_array = np.round(mean_array) - 2
            mean_array = np.clip(mean_array, 0, np.max(mean_array))
            return_dict[chrom] = np.triu(mean_array)
        return return_dict

    elif infer_task in [6, 7]:
        #embedding generation
        return_dict={"submat_embedding":{},"patch_embedding":{},"chromo_embedding":{},"genome_embedding":{}}
        return_dict["skipped_regions"] = output_dict["skipped_regions"]

        #read patch embedding in output_dict, average the same location embedding
        for key in output_dict["patch_embedding"]:
            cur_embedding = output_dict["patch_embedding"][key]
            cur_embedding = np.stack(cur_embedding,axis=0)
            cur_embedding = np.mean(cur_embedding,axis=0)
            return_dict["patch_embedding"][key] = cur_embedding
        
        #read submat embedding in output_dict, average the same location embedding
        chrom_embedding = defaultdict(list)
        for key in output_dict["submat_embedding"]:
            cur_embedding = output_dict["submat_embedding"][key]
            cur_embedding = np.stack(cur_embedding,axis=0)
            cur_embedding = np.mean(cur_embedding,axis=0)
            return_dict["submat_embedding"][key] = cur_embedding
            chrom = key.split(":")[0]
            chrom_embedding[chrom].append(cur_embedding)
        
        #get average chromo embedding
        for chrom in chrom_embedding:
            cur_embedding = chrom_embedding[chrom]
            cur_embedding = np.stack(cur_embedding,axis=0)
            cur_embedding = np.mean(cur_embedding,axis=0)
            return_dict["chromo_embedding"][chrom] = cur_embedding
        #get average genome embedding
        all_embedding = list(return_dict["chromo_embedding"].values())
        all_embedding = np.stack(all_embedding,axis=0)
        all_embedding = np.mean(all_embedding,axis=0)
        return_dict["genome_embedding"] = all_embedding
        return return_dict