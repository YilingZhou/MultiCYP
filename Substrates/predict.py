from chemprop.utils import load_checkpoint
import pandas as pd
from chemprop.args import TrainArgs
from chemprop.data import get_data, get_task_names, MoleculeDataLoader
from chemprop.train import predict
from .units import cal_metrics, get_pd_mean_var, get_pd_mean_var2
from .load_model import multi_model

tasks = ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']

def save_pres(test_data, avg_test_preds,args):

    test_preds_dataframe = pd.DataFrame(data={'Smiles': [i[0] for i in test_data.smiles()]})

    for i, task_name in enumerate(args.task_names):
        test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

    return test_preds_dataframe

def multi_predict(modelpath, datapath):

    # init args
    arguments = ['--data_path', './train2.csv',
                 '--dataset_type', 'classification']
    args = TrainArgs().parse_args(arguments)

    # set
    args.epochs = 30
    args.depth = 3
    args.batch_size = 1
    args.hidden_size = 300

    args.data_path = datapath
    args.target_columns = ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']
    args.smiles_columns = ['Smiles']
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    # load model
    model = load_checkpoint(modelpath)

    # get data
    test_data = get_data(path=args.data_path, args=args)
    test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=1, num_workers=0)

    # predict
    test_preds = predict(model=model, data_loader=test_data_loader)
    pred_pd = save_pres(test_data, test_preds, args)
    return pred_pd

def ensemble_pred(modelpath_list, datapath, varout=False):

    pred_pd_list = []
    for modelpath in modelpath_list:
        pred_pd = multi_predict(modelpath, datapath)
        pred_pd_list.append(pred_pd)

    mean_pd, var_pd = get_pd_mean_var2(pred_pd_list)

    if varout:
        return mean_pd, var_pd
    else:
        return mean_pd

def get_pred_label(pred_pd, threshold=0.5):
    for col in tasks:
        pred_pd[col] = pred_pd[col].apply(lambda x: 1 if x > threshold else 0)
    return pred_pd

if __name__ == '__main__':
    print('')
    # pred_pd = multi_predict(modelpath=multi_model[0], datapath='./test2.csv')
    # label_pd = get_pred_label(pred_pd)
    # label_pd.to_csv('./try.csv', index=False)
# add avg*

    pred_pd = ensemble_pred(modelpath_list=multi_model, datapath='./cypreact_test.csv')
    label_pd = get_pred_label(pred_pd)
    label_pd.to_csv('./cyreact_test_label.csv', index=True)
    print(1)
# metrics
    test = pd.read_csv('cypreact_test.csv', index_col=False)
    test_pred = pd.read_csv('./cyreact_test_label.csv', index_col=False)
    metrics = cal_metrics(test=test, test_pred=test_pred)
    metrics.to_csv('cyprect_metrics.csv', index=True)


