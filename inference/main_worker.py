import torch
import os
import torch.nn as nn
import pandas as pd
from scipy.sparse import coo_matrix

from utils.hic_coverage import calculate_coverage
from data_processing.inference_dataset import Inference_Dataset
from ops.io_utils import write_pickle,append_record
from ops.mean_shift_merge import mean_shift_merge
from ops.file_format_convert import pkl2others
from utils.array2bigwig import array2bigwig
from model.pos_embed import interpolate_pos_embed_inputsize


def configure_dataset(args,input_pkl):
    resolution = args.resolution
    import torchvision.transforms as transforms
    transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.task==3:
        #resolution enhancement
        fill_diagonal_zero=True
        
    else:
        fill_diagonal_zero=False
    if args.task==3:
        #judge if it is a very deep sequencing data, if it is, set max_cutoff to None
        coverage_perresolution = calculate_coverage(input_pkl)/resolution
        if coverage_perresolution>1:
            max_cutoff = None
        else:
            max_cutoff = 100
    elif args.task==2:
        #loop calling
        max_cutoff = 1000
    elif args.task==5:
        #scHi-C enhancement
        max_cutoff = 100
    else:
        max_cutoff = None
    
    if args.task==4:
        #epigenomic assay prediction
        locus_embedding = True
    else:
        locus_embedding = False
        
    bounding = args.bound
    stride = args.stride
    input_row_size = args.input_row_size
    input_col_size = args.input_col_size
    task = args.task
    input_bedpe = args.bedpe_path
    resolution = args.resolution
    dataset = Inference_Dataset(data_path=input_pkl,
                            bedpe_path=input_bedpe,
                            resolution=resolution,    
                            transform=transform_input,
                            stride=stride,
                            window_height= input_row_size,
                            window_width = input_col_size,
                            max_cutoff=max_cutoff,
                            fill_diagonal_zero=fill_diagonal_zero,
                            bounding=bounding,
                            locus_embedding=locus_embedding,
                            task=task)
    sample_batch_size = args.batch_size
    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    return data_loader_test

def generate_loop(return_dict,threshold,output_bedpe,config_resolution):
    with open(output_bedpe,'w') as wfile:
        wfile.write("chr1\tx1\tx2\tchr2\ty1\ty2\n")
    for chrom in return_dict:
        mean_array = return_dict[chrom]
        if mean_array.shape[0]<=10000:
            mean_array = mean_array.toarray()
            try:
                mean_loc_list = mean_shift_merge(mean_array,cutoff=threshold)
            except:
                mean_loc_list = []
        else:
            mean_loc_list = []
            for i in range(0,mean_array.shape[0],10000):
                cur_start = i
                cur_end = min(i+10000,mean_array.shape[0])
                select_index1 = (mean_array.row>=cur_start)&(mean_array.row<cur_end)
                select_index2 = (mean_array.col>=cur_start)&(mean_array.col<cur_end)
                select_index = select_index1&select_index2
                cur_select_row = mean_array.row[select_index]-cur_start
                cur_select_col = mean_array.col[select_index]-cur_start
                cur_select_data = mean_array.data[select_index]
                cur_size = cur_end-cur_start
                cur_array = coo_matrix((cur_select_data,(cur_select_row,cur_select_col)),shape=(cur_size,cur_size))
                cur_array = cur_array.toarray()
                try:
                    cur_loc_list = mean_shift_merge(cur_array,cutoff=threshold)
                except:
                   cur_loc_list = []
                for loc in cur_loc_list:
                    x,y = loc
                    x+=cur_start
                    y+=cur_start
                    mean_loc_list.append([x,y])
                print(i, "detect length: mean",len(cur_loc_list),"total",len(mean_loc_list))
       
        print("%s detect length: mean"%chrom,len(mean_loc_list))
        if "_" in chrom:
            chrom = chrom.split("_")[0]
        append_record(output_bedpe,mean_loc_list,chrom,resolution=config_resolution)
def main_worker(args, input_pkl):
    resolution = args.resolution
    #check model_path exists
    model_path = os.path.abspath(args.model_path)
    assert os.path.exists(model_path), "model_path does not exist"
    output_dir = os.path.abspath(args.output)
    dataloader = configure_dataset(args, input_pkl)
    import model.Vision_Transformer_count as Vision_Transformer
    #should be a dyanmic input model
    patch_wise_size = (args.input_row_size//args.patch_size,args.input_col_size//args.patch_size)
    vit_backbone = Vision_Transformer.__dict__[args.model](img_size=(args.input_row_size,args.input_col_size))
    if args.task in [6, 7]:
        # embedding genration inference
        # only load encoder weights
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = vit_backbone.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        #this can apply to most scenarios but not our condition
        
        interpolate_pos_embed_inputsize(vit_backbone, checkpoint_model,input_size=patch_wise_size,
                                            use_decoder=False)
        # load pre-trained model
        msg = vit_backbone.load_state_dict(checkpoint_model, strict=False)
        print("Loading pre-train encoder message:",msg)
    from model.Finetune_Model_Head import Finetune_Model_Head
    model = Finetune_Model_Head(vit_backbone, task=args.task,
                            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                        mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=patch_wise_size)
    
    
    #load model weights
    if args.task not in [6, 7]:
        checkpoint = torch.load(model_path, map_location='cpu')
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        else:
            checkpoint_model = checkpoint
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Loading fine-tuned task-specific model message:",msg)
    else:
        
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        #loading pre-trained decoder
        interpolate_pos_embed_inputsize(model, checkpoint['model'],
                                        input_size=patch_wise_size,use_decoder=True)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Loading pre-train model decoder message:",msg)

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    from inference.inference_worker import inference_worker
    return_dict= inference_worker(model,dataloader,
                                  log_dir=output_dir,
                                  args=args)
    if args.task==1:
        output_path = os.path.join(output_dir,"HiCFoundation_reproducibility_embedding.pkl")
        write_pickle(return_dict,output_path)
        print("Reproducibility analysis finished!")
        print("The embedding results are saved to ",output_path)
    elif args.task==2:
        #0.9 is used for benchmark, but please choose the threshold based on your own data

        threshold_list= [0.5,0.75,0.9]
        for threshold in threshold_list:
            output_bedpe = os.path.join(output_dir,"HiCFoundation_loop_{}.bedpe".format(threshold))
            generate_loop(return_dict,threshold,output_bedpe,resolution)
        print("Loop calling finished!")
        print("The loop calling results are saved to ",output_dir," with different thresholds in .bedpe format.")
    elif args.task==3:
        #convert to hic format as final output
        output_pkl = os.path.join(output_dir,"HiCFoundation_enhanced.pkl")
        #revise the return dict key if it has "_", make to one chromosome
        for key in list(return_dict.keys()):
            if "_" in key:
                key_list = key.split("_")
                return_dict[key_list[0]] = return_dict[key]
                del return_dict[key]
        write_pickle(return_dict,output_pkl)
        input_file = os.path.abspath(args.input)
        extention_name = input_file.split('.')[-1]
        output_file = os.path.join(output_dir,"HiCFoundation_enhanced."+extention_name)
        pkl2others(output_pkl, output_file,resolution,args.genome_id)
        if not os.path.exists(output_file):
            print("Error: file conversion failed.")
            print("Resolution enhancement finished!")
            print("The final output is saved in .pkl format, please convert it to other formats manually.")
            print("The .pkl file is saved to ",output_pkl)
    elif args.task==4:
        #epigenomic assay prediction
        output_path = os.path.join(output_dir,"HiCFoundation_epigenomic_assay_prediction.pkl")
        write_pickle(return_dict,output_path)
        #write to bigWig file
        key_word_list=['CTCF','H3K4me3','H3K27ac','H3K27me3','ATAC-seq','DNase-seq']
        for key_index,key_word in enumerate(key_word_list):
            current_dict={}
            for chrom in return_dict:
                if "_" in chrom:
                    chrom_key = chrom.split("_")[0]
                else:
                    chrom_key = chrom
                current_dict[chrom_key] = return_dict[chrom][key_index]
            current_pkl = os.path.join(output_dir,"HiCFoundation_epigenomic_assay_prediction_%s.pkl"%key_word)
            write_pickle(current_dict,current_pkl)
            output_bigwig = os.path.join(output_dir,"HiCFoundation_pred_%s.bigWig"%key_word)
            array2bigwig(current_pkl,output_bigwig,resolution=resolution)
        print("Epigenomic assay prediction finished!")
        print("The prediction results are saved to ",output_dir," in .pkl and .bigWig format.")

    elif args.task==5:
        #scHi-C enhancement
        output_path = os.path.join(output_dir,"HiCFoundation_sc_enhanced.pkl")
        write_pickle(return_dict,output_path)
        input_file = os.path.abspath(args.input)
        extention_name = input_file.split('.')[-1]
        output_file = os.path.join(output_dir,"HiCFoundation_sc_enhanced."+extention_name)
        pkl2others(output_path, output_file,resolution,args.genome_id)
        if not os.path.exists(output_file):
            print("Error: file conversion failed.")
            print("scHi-C enhancement finished!")
            print("The final output is saved in .pkl format, please convert it to other formats manually.")
            print("The .pkl file is saved to ",output_path)
    elif args.task in [6, 7]:  
        #embedding generation
        output_skipped_path = os.path.join(output_dir,"skipped_regions.bedpe")
        skipped_df = pd.DataFrame(return_dict["skipped_regions"], columns=["chrom1", "row_start", "row_end", "chrom2", "col_start", "col_end"])
        skipped_df.to_csv(output_skipped_path, index = False, header = None, sep = "\t")
        
        output_path = os.path.join(output_dir,"HiCFoundation_embedding.pkl")
        return_dict_new = {}
        return_dict_new["submat_embedding"] = return_dict["submat_embedding"]
        return_dict_new["patch_embedding"] = return_dict["patch_embedding"]
        return_dict_new["chromo_embedding"] = return_dict["chromo_embedding"]
        return_dict_new["genome_embedding"] = return_dict["genome_embedding"]
        write_pickle(return_dict_new,output_path)
        print("Hi-C embedding generation finished!")
        print("The embedding results are saved to ",output_path," in .pkl format.")


    print("Enjoy your HiCFoundation results!")