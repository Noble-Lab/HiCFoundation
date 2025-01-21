import argparse

def argparser_infer():
    parser = argparse.ArgumentParser('HiCFoundation inference', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size of the input')
    # Dataset parameters
    parser.add_argument('--input', type=str, help='a .hic/.cool/.pkl/.txt/.pairs/.npy file records Hi-C/scHi-C matrix')
    parser.add_argument('--resolution', default=10000, type=int,help='resolution of the input matrix')
    parser.add_argument("--task",default=0,type=int,help="1: Reproducibility analysis; \n 2: Loop calling; \n \
                        3: Resolution enhancement; \n 4: Epigenomic assay prediction; \n 5: scHi-C enhancement \n \
                        6: Hi-C embedding generation for all regions \n 7. Hi-C embedding generation for specified regions")
    
    parser.add_argument('--input_row_size', default=224, type=int,
                        help='input submatrix row size')
    parser.add_argument("--input_col_size",default=4000,type=int,help="input submatrix column size")
    parser.add_argument("--patch_size",default=16,type=int,help="patch size for the input submatrix")

    parser.add_argument('--stride', default=20, type=int,
                        help='scanning stride for the input Hi-C matrix')
    parser.add_argument("--bound",default=200,type=int,help="off-diagonal bound for the scanning")
    parser.add_argument('--num_workers', default=8, type=int,help="data loading workers per GPU")
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--model_path",default='',help='load fine-tuned model for inference')
    parser.add_argument('--output', default='hicfoundation_inference',help='output directory to save the results')
    parser.add_argument("--print_freq",default=1,type=int,
                        help="log frequency for output log during inference")
    parser.add_argument("--gpu",default=None,type=str,help="which gpu to use")
    parser.add_argument("--genome_id",type=str,default="hg38", help="genome id for generating .hic file. \n \
                        Must be one of hg18, hg19, hg38, dMel, mm9, mm10, anasPlat1, bTaurus3, canFam3, equCab2, \
                        galGal4, Pf3D7, sacCer3, sCerS288c, susScr3, or TAIR10; \n \
                         alternatively, this can be the path of the chrom.sizes file that lists on each line the name and size of the chromosomes.")
    parser.add_argument("--embed_depth",default=0,type=int,help="0: embedding from encoder; k: embedding from k-th layer of decoder; up to 8 indicating the final output of decoder.")
    parser.add_argument("--patch_embedding", action='store_true',
                        help='Record the patch embedding in the final output file. Default: False')
    # argument only relevant for task 7: region-specific embedding
    parser.add_argument('--bedpe_path', default="", type=str, 
                        help='a .bedpe file specifying the rectangular genomic regions for extracting HiCfoundation embeddings (used only in Task 7)')
   
    return parser

def argparser_finetune():
    parser = argparse.ArgumentParser('HiCFoundation fine-tuning', add_help=False)
    #config fine-tuning settings
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size of the input')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)\n \
                        The effective batch size is batch_size*accum_iter. \n If you have memory constraints, \
                        you can increase --accum_iter and reduce the --batch_size to trade off memory for computation. \n \
                        ')
    parser.add_argument("--epochs",default=50,type=int,help="number of epochs for fine-tuning")
    parser.add_argument("--warmup_epochs",default=5,type=int,
                        help="number of warmup epochs for fine-tuning")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch of fine-tuning. \n \
                            It will be used for resuming training and automatically load from the checkpoint.')
    
    # configure optimizer settings
    parser.add_argument("--lr",default=None,type=float,help="learning rate for fine-tuning. This should not be set, \n \
                        it will be calculated through --batch_size and --blr.")
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for learning rate decay during fine-tuning.')
    parser.add_argument("--weight_decay",default=0.05,type=float,help="weight decay for fine-tuning")
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay during fine-tuning')
    
    # configure the fine-tuning model settings
    parser.add_argument("--model",default='vit_large_patch16',
                        type=str,help="model name for fine-tuning")
    parser.add_argument("--pretrain",default='',type=str, help='load pre-trained model for fine-tuning')
    #add resume to support resuming training from a checkpoint
    parser.add_argument("--resume",default='',type=str,help='resume fine-tuning from a checkpoint')
    parser.add_argument("--finetune",default=0,type=int,help="1: only fine-tune the model's encoder; 2: fine-tune the whole model;")
    parser.add_argument('--seed', default=888, type=int,help="random seed for fine-tuning. It is used to make sure results are reproducible.")
    #configure loss type
    parser.add_argument("--loss_type",default=0,type=int,help="1: MSE loss; 2: Cosine loss; \n You can define your own loss function in finetune/loss.py")

    # Dataset parameters
    parser.add_argument('--data_path', type=str, help='a directory contains many sub-directory, each sub-dir includes many .pkl files for fine-tuning. \n \
                        The .pkl file should record a dict with following keys refer to different fine-tuning purposes\n \
                        "input": the input Hi-C/scHi-C matrix in scipy.sparse or numpy.array format; \n \
                        "input_count": the total count of Hi-C expriment (optional); \n \
                        "2d_target": the output Hi-C/scHi-C matrix in scipy.sparse or numpy.array format; \n \
                        "embed_target": the embedding vector in numpy.array format; \n \
                        "1d_target": the 1D target vector in numpy.array format; \n \
                        The last three keys are optional, you can adjust it based on your fine-tuning purpose.')
    parser.add_argument("--train_config",type=str,help="a .txt file records the training information for input directory. \n \
                        Each line should be the sub-dir name that will be used to train during fine-tuning. \n ")
    parser.add_argument("--valid_config",type=str,help="a .txt file records the validation information for input directory. \n \
                        Each line should be the sub-dir name that  will be used to validate during fine-tuning. \n ")
    #add output config
    parser.add_argument('--output', default='hicfoundation_finetune',help='output directory to save the results. \n \
                        The output directory will contain the fine-tuned model, log files, and tensorboard logs.')
    parser.add_argument("--tensorboard",default=0,type=int,help="1: enable tensorboard log for fine-tuning; 0: disable tensorboard log for fine-tuning")

    #config distributed training 
    parser.add_argument('--device', default='cuda', help='device to use for fine-tuning iterations')
    parser.add_argument('--num_workers', default=8, type=int,help="workers per GPU. You can adjust it based on your server's configuration.")
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    #parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,help="specify the rank of the server, default 0.")

    #configure input size
    parser.add_argument('--input_row_size', default=224, type=int,
                        help='input submatrix row size')
    parser.add_argument("--input_col_size",default=4000,type=int,help="input submatrix column size")
    parser.add_argument("--patch_size",default=16,type=int,help="patch size for the input submatrix")
    
    #configure print/save frequency
    parser.add_argument("--print_freq",default=1,type=int,
                        help="log frequency for output log during fine-tuning")
    parser.add_argument("--save_freq",default=1,type=int,
                        help="save frequency for saving the fine-tuned model")
    parser.add_argument("--gpu",default="0",type=str,help="which gpu to use, will be configured by the script automatically")


    return parser


def  argparser_pretrain():
    parser = argparse.ArgumentParser('HiCFoundation pre-training', add_help=False)
    #config pre-training settings
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size of the input')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)\n \
                        The effective batch size is batch_size*accum_iter. \n If you have memory constraints, \
                        you can increase --accum_iter and reduce the --batch_size to trade off memory for computation. \n \
                        ')
    parser.add_argument("--epochs",default=100,type=int,help="number of epochs for pre-training")
    parser.add_argument("--warmup_epochs",default=10,type=int,
                        help="number of warmup epochs for pre-training")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch of pre-training. \n \
                            It will be used for resuming training and automatically load from the checkpoint.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument("--sparsity_ratio",default=0.05,type=float,
                        help="Used to the submatrix if the valid contact is less than sparsity_ratio*region_size")
    parser.add_argument("--loss_alpha",default=1,type=float,help="loss weight for other losses to combine with the patch-contrastive loss")
    
    # configure optimizer settings
    parser.add_argument("--lr",default=None,type=float,help="learning rate for fine-tuning. This should not be set, \n \
                        it will be calculated through --batch_size and --blr.")
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for learning rate decay during fine-tuning.')
    parser.add_argument("--weight_decay",default=0.05,type=float,help="weight decay for fine-tuning")

    
    # configure the fine-tuning model settings
    parser.add_argument("--model",default='vit_large_patch16',
                        type=str,help="model name for fine-tuning")
    #add resume to support resuming training from a checkpoint
    parser.add_argument("--resume",default='',type=str,help='resume fine-tuning from a checkpoint')
    parser.add_argument('--seed', default=888, type=int,help="random seed for fine-tuning. It is used to make sure results are reproducible.")
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, help='a directory contains many sub-directory, each sub-dir includes many .pkl files for pre-training. \n \
                        The .pkl file should record a dict with following keys refer to different fine-tuning purposes\n \
                        "input": the input Hi-C/scHi-C matrix in scipy.sparse or numpy.array format; \n \
                        "input_count": the total count of corresponding Hi-C expriment (optional); \n \
                        "diag": the diagonal starting index for the input Hi-C matrix; \n \
                        If it is smaller than 0, it indicates the diagonal starts at (diag,0) position; \n \
                        If it is larger than 0, it indicates the diagonal starts at (0,diag) position; \n \
                        If its absolute value is larger than the matrix size, it indicates no diagonal info here. \n \
                        You can also specify "diag" as None or do not include this info to indicate no diag to consider here ')
    parser.add_argument("--train_config",type=str,help="a .txt file records the training information for input directory. \n \
                        Each line should be the sub-dir name that will be used to train during fine-tuning. \n ")
    parser.add_argument("--valid_config",type=str,help="a .txt file records the validation information for input directory. \n \
                        Each line should be the sub-dir name that  will be used to validate during fine-tuning. \n ")
    #add output config
    parser.add_argument('--output', default='hicfoundation_finetune',help='output directory to save the results. \n \
                        The output directory will contain the fine-tuned model, log files, and tensorboard logs.')
    parser.add_argument("--tensorboard",default=0,type=int,help="1: enable tensorboard log for fine-tuning; 0: disable tensorboard log for fine-tuning")

    #config distributed training 
    parser.add_argument('--device', default='cuda', help='device to use for fine-tuning iterations')
    parser.add_argument('--num_workers', default=8, type=int,help="workers per GPU. You can adjust it based on your server's configuration.")
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    #parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,help="specify the rank of the server, default 0.")

    #configure input size
    parser.add_argument('--input_row_size', default=224, type=int,
                        help='input submatrix row size')
    parser.add_argument("--input_col_size",default=4000,type=int,help="input submatrix column size")
    parser.add_argument("--patch_size",default=16,type=int,help="patch size for the input submatrix")
    
    #configure print/save frequency
    parser.add_argument("--print_freq",default=1,type=int,
                        help="log frequency for output log during fine-tuning")
    parser.add_argument("--save_freq",default=1,type=int,
                        help="save frequency for saving the fine-tuned model")
    parser.add_argument("--gpu",default="0",type=str,help="which gpu to use, will be configured by the script automatically")


    return parser