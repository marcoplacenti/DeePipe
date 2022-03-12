from src.pipe.MLPipe import MLPipe

import argparse
import yaml

def parse_config(args):
    config_file = args.config
    with open(config_file) as infile:
        config_dict = yaml.load(infile, Loader=yaml.SafeLoader)
    return config_dict


def run(config_dict):
    pipe = MLPipe(config_dict)
    pipe.preproc_data()
    pipe.train()
    #pipe.eval()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Pipe NN.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    config_dict = parse_config(args)

    run(config_dict)
    
    
