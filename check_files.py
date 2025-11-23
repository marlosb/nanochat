"""
Check the integrity of parquet files in the dataset directory.
"""
import os
import pyarrow.parquet as pq
from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files

def check_parquet_integrity():
    """Check all parquet files for corruption."""
    parquet_paths = list_parquet_files()
    
    if not parquet_paths:
        print("No parquet files found!")
        return
    
    print(f"Found {len(parquet_paths)} parquet file(s)")
    print("-" * 80)
    
    corrupted_files = []
    valid_files = []
    
    for filepath in parquet_paths:
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)
        filesize_mb = filesize / (1024 * 1024)
        
        try:
            # Try to open the parquet file
            pf = pq.ParquetFile(filepath)
            num_rows = pf.metadata.num_rows
            num_row_groups = pf.num_row_groups
            
            # Try to read the first row group to ensure it's readable
            if num_row_groups > 0:
                first_rg = pf.read_row_group(0)
                
            print(f"✓ {filename}")
            print(f"  Size: {filesize_mb:.2f} MB")
            print(f"  Rows: {num_rows:,}")
            print(f"  Row groups: {num_row_groups}")
            valid_files.append(filepath)
            
        except Exception as e:
            print(f"✗ {filename}")
            print(f"  Size: {filesize_mb:.2f} MB")
            print(f"  ERROR: {str(e)}")
            corrupted_files.append(filepath)
        
        print()
    
    # Summary
    print("-" * 80)
    print(f"Summary:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nCorrupted files that should be deleted/re-downloaded:")
        for filepath in corrupted_files:
            print(f"  - {filepath}")
        print(f"\nTo delete corrupted files, run:")
        for filepath in corrupted_files:
            print(f"  rm \"{filepath}\"")
        
        # Also check for .tmp files
        base_dir = get_base_dir()
        data_dir = os.path.join(base_dir, "base_data")
        tmp_files = [f for f in os.listdir(data_dir) if f.endswith('.tmp')]
        if tmp_files:
            print(f"\nTemporary files found (should also be deleted):")
            for tmp_file in tmp_files:
                tmp_path = os.path.join(data_dir, tmp_file)
                print(f"  rm \"{tmp_path}\"")
    
    return corrupted_files

if __name__ == "__main__":
    check_parquet_integrity()