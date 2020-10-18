import data
from classification import Classification

if __name__ == "__main__":
    cl = Classification(data.load_classes())
    cl.process(data.load_samples())
