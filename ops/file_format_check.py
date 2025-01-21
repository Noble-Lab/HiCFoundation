import os
import pandas as pd

def check_bedpe(args, require_intra_chrom):
    """
    This helper function validats a .bedpe file
    based on specified criteria and separates
    valid and invalid rows.
    
    Parameters:
    args: Parsed command-line arguments containing:
        - bedpe_path (str): path to the input .bedpe file
        - resolution (int): resolution of the input matrix
        - input_row_size (int): expected size for (end1 - start1) // args.resolution
        - input_col_size (int): expected size for (end2 - start2) // args.resolution
        - output (str): directory where output files will be saved.
    require_intra_chrom (bool): 
        - If True, requires all rows to specify intra-chromosomal regions
        - If False, there is no requirement on chromosomal regions
    
    Returns:
        str: full path to the valid_input.bedpe file if valid rows exist, else None.
    """
    expected_columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    
    input_bedpe = os.path.abspath(args.bedpe_path)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok = True)
    
    valid_bedpe_path = os.path.join(output_dir, "valid_input.bedpe")
    invalid_bedpe_path = os.path.join(output_dir, "invalid_input.bedpe")
    
    try:
        df = pd.read_csv(
            input_bedpe,
            sep = '\t',
            header = None,
            usecols = range(6),
            names = expected_columns,
            dtype = {
                "chrom1": str,
                "start1": int,
                "end1": int,
                "chrom2": str,
                "start2": int,
                "end2": int
                },
            # if it contain bad lines with weird fields,
            # throw errors directly 
            on_bad_lines = "error"
            )
    except FileNotFoundError:
        print(f"Error the file {input_bedpe} does not exist.")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file {input_bedpe} is empty or does not contain valid data.")
        return None
    except pd.errors.ParserError as e:
        print(f"Parser error while reading {input_bedpe}: {e}.")
        return None
    except Exception as e:
        print(f"Error reading the .bedpe file: {e}")
        return None
    
    # check for negative coordinate values
    valid_coords = (
        (df["start1"] >= 0) & (df["end1"] >= 0) 
        & (df["start2"] >= 0) & (df["end2"] >= 0)
        )
    
    # check for valid row size and col size
    valid_size_row = (df["end1"] - df["start1"]) == (args.input_row_size * args.resolution)
    valid_size_col = (df["end2"] - df["start2"]) == (args.input_col_size * args.resolution)
    
    # check if rows are intra-chromosomal regions (currently only support intra-chromosomal)
    is_intra_chrom = (df["chrom1"] == df["chrom2"])
    if require_intra_chrom:
        valid_rows = valid_coords & valid_size_row & valid_size_col & is_intra_chrom
    else:
        valid_rows = valid_coords & valid_size_row & valid_size_col
    invalid_rows = ~valid_rows
    
    df_valid = df[valid_rows]
    df_invalid = df[invalid_rows]
    
    if df_valid.shape[0] == 0:
        print(f"Error input .bedpe file contains 0 valid rows. Skipping!")
        return None
        
    try:
        df_valid.to_csv(valid_bedpe_path, sep = "\t", header = None, index = False)
        print(f"Valid rows saved to: {valid_bedpe_path}")
        df_invalid.to_csv(invalid_bedpe_path, sep = "\t", header = None, index = False)
        print(f"Invalid rows saved to: {invalid_bedpe_path}")
    
    except Exception as e:
        print(f"Error writing the output files: {e}")
        return None

    return valid_bedpe_path