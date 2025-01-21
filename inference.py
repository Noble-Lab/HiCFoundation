import os
import timm
assert timm.__version__ == "0.3.2" # version check for timm
from ops.argparser import  argparser_infer
from ops.file_format_convert import convert_to_pkl

def main(args):
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("local ip: ",local_ip)
    #format processing, convert different formats to .pkl format for further processing
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir,exist_ok=True)
    input_file = os.path.abspath(args.input)
    config_resolution = args.resolution
    input_pkl=convert_to_pkl(input_file, output_dir,config_resolution)
    
    #for reproducibility analysis, we need to smooth the matrix to generate embeddings.
    if args.task==1:
        from ops.smooth_matrix import smooth_pkl
        smooth_pkl_file = os.path.join(output_dir,"input_smoothed.pkl")
        input_pkl = smooth_pkl(input_pkl,smooth_pkl_file)
        print("Reproducibility analysis smoothed input matrix saved to ",input_pkl)
    if args.task==7:
        from ops.file_format_check import check_bedpe
        # Helper function to create a new .bedpe file 
        # containing all valid rows from the original .bedpe file
        valid_bedpe_path = check_bedpe(args, require_intra_chrom=True)
        if valid_bedpe_path:
            args.bedpe_path = valid_bedpe_path
            print("Valid bedpe_path is at ", args.bedpe_path)
        else:
            print("Error in processing bedpe file. Task 7 terminated.")
            exit(1)
            
    from inference.main_worker import main_worker
    main_worker(args, input_pkl)


if __name__ == '__main__':
    print("HiCFoundation inference started!")
    parser = argparser_infer()
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #print mode based on --task
    if args.task==1:
        print("Reproducibility analysis")
    elif args.task==2:
        print("Loop calling")
    elif args.task==3:
        print("Resolution enhancement")
    elif args.task==4:
        print("Epigenomic assay prediction")
    elif args.task==5:
        print("scHi-C enhancement")
    elif args.task==6:
        print("Hi-C embedding generation for all regions")
        embed_depth = args.embed_depth
        if embed_depth>8:
            print("Error: embed_depth is larger than 8, that is beyond decoder depth. Please set embed_depth<=8")
            print("0 indicates the encoder output, k indicates the k-th decoder layer's output")
            exit(1)
    elif args.task==7:
        print("Hi-C embedding generation for specified regions")
        bedpe_path = args.bedpe_path
        if not bedpe_path.endswith(".bedpe"):
            print("Error: needs to input a .bedpe file")
            exit(1)
    else:
        print("Unknown task specified ",args.task)
        print("Please specify the task using --task with 1,2,3,4,5,6")
        exit(1)
    #check the specied input size, must be a multiple of args.patch_size
    if args.input_row_size%args.patch_size!=0 or args.input_col_size%args.patch_size!=0:
        print("args configuration error: input_row_size and input_col_size must be a multiple of patch_size")
        exit(1)
    #output the args in a beautiful format
    main(args)

