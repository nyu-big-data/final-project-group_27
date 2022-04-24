
import sys
import json

#Main Function that will define model behavior
def main(model_size,model_type,model_args):

    print(model_size, model_type, model_args)


if __name__ == '__main__':
    """
    Model argumensts hsould be entered in the following order:
    1) model size -> either "small" or "large"
    2) model type -> either "baseline" or "als" (More to Do)
    3) Dictionary of {parameter:argument} pairs that will be parsed to the model
    i.e. '{"rank":10, "maxIter":10,"regParam":0.05}'
    """
    #Model size is either "small" or "large"
    model_size = sys.argv[1]+"_"
    #Define the model type in second argument:
    model_type = sys.argv[2]
    #Model Args:
    params = sys.argv[3]
    model_args = json.loads(params)
    main(model_size,model_type,model_args)