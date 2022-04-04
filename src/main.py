from src.pipe.MLPipe import MLPipe

import argparse

def run(config_dict):
    pipe = MLPipe(config_dict)
    pipe.preproc_data()
    pipe.train()
    pipe.eval()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Pipe NN.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    run(args.config)
    
    
