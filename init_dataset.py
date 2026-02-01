import os, sys, logging, argparse, shutil, subprocess, shlex

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Initialize dataset for fingerprint detection.")
    parser.add_argument("--dataset-path", default="/green/data/L3SF_V2/L3-SF/", help="Path to the dataset.")
    parser.add_argument("--dataset-name", default="L3-SF", help="Name of the dataset.")
    parser.add_argument("--conda-env", default="detection", help="Conda env name.")
    args = parser.parse_args()

    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info("Starting database initialization process.")

    repo_root = os.path.dirname(__file__)

    pore_dir  = os.path.join(repo_root, "pore-detection")

    cmd = [
        "conda", "run", "-n", "detection", "python",
        os.path.join(pore_dir, "pore_detect_clean.py"),
        "--dataset-path", args.dataset_path,
        "--dataset-name", args.dataset_name,
    ]
   

    logger.info("Running detection: %s", " ".join(map(shlex.quote, cmd)))
    try:
        subprocess.run(cmd, check=True, cwd=pore_dir)
        logger.info("Pore detection subprocess completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Pore detection failed (rc={e.returncode}).")
        sys.exit(1)

    src_dir = os.path.join(repo_root, "pore-detection", "out_of_the_box_detect", "Prediction", args.dataset_name)
    dst_dir = os.path.join(repo_root, "dataset", args.dataset_name)
    os.makedirs(dst_dir, exist_ok=True)

    try:
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dst_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        logger.info(f"Copied files from {src_dir} to {dst_dir}")
    except Exception as e:
        logger.error(f"Failed to copy files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
