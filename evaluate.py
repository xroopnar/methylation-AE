import utils
import model
import shap

def evaluate_holdout():
    pass


def main():
    model = run_go()
    X,y = utils.fetch_go_xy()
    background = X.sample(100)
    e = shap.DeepExplainer(model,background)
    return(e)
