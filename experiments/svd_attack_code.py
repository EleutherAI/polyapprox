import matplotlib.pyplot as plt
import os
from experiments.functional_plotting_utils import svd_attack_projection, evaluate_model, ckpt_to_model, startup

import torch

recreate_data = False
overwrite_datadict = False
create_plot = False
savefig = False

def main():

    '''
    Reproduces SVD attack data.
    '''


    if recreate_data:


        datadict_path = f'/Users/alicerigg/Code/polyapprox/experiments/data/baseline_centeredlong_datadict.pt'

        datadict, ckpt, config, dataset, mu, scale, model, ols = startup(datadict_path)
        svd_attack_data = {}
        print(datadict.keys())

        for idx in range(len(datadict['ckpts'])):
            ckpt = datadict['ckpts'][idx]
            ckpt_model = ckpt_to_model(ckpt, config)
            linear_ols = ckpt_model.approx_fit('linear')

            svd_attack_data[idx] = {'relu': {}, 'linear': {}}
            svd_attack_relu_accs = []
            svd_attack_relu_logits = []
            svd_attack_linear_accs = []
            svd_attack_linear_logits = []

            for topk in range(11):
                proj = svd_attack_projection(linear_ols.beta, topk=topk)
                linear_fwd, linear_acc = evaluate_model(linear_ols, dataset, proj=proj)
                model_fwd, model_acc = evaluate_model(ckpt_model, dataset, proj=proj)
                print(idx, topk, linear_acc, model_acc)
                svd_attack_relu_logits.append(model_fwd)
                svd_attack_relu_accs.append(model_acc)
                svd_attack_linear_logits.append(linear_fwd)
                svd_attack_linear_accs.append(linear_acc)

            svd_attack_relu_accs = torch.stack([torch.tensor(acc) for acc in svd_attack_relu_accs])
            svd_attack_relu_logits = torch.stack(svd_attack_relu_logits)
            svd_attack_linear_accs = torch.stack([torch.tensor(acc) for acc in svd_attack_linear_accs])
            svd_attack_linear_logits = torch.stack(svd_attack_linear_logits)

            datadict['plotting_data']['svd_attack'] = {}
            num_ckpts = len(datadict['ckpts'])
            

            svd_attack_data[idx]['relu']['accuracy'] = svd_attack_relu_accs
            svd_attack_data[idx]['relu']['testset_logits'] = svd_attack_relu_logits
            svd_attack_data[idx]['linear']['accuracy'] = svd_attack_linear_accs
            svd_attack_data[idx]['linear']['testset_logits'] = svd_attack_linear_logits

    if overwrite_datadict:
        datadict['plotting_data']['svd_attack'] = svd_attack_data
        torch.save(datadict, datadict_path)

    if create_plot:
        idx = 12
        # Plotting SVD attack data for relu vs linear traces on the 'accuracy' key
        relu_accuracy = svd_attack_data[idx]['relu']['accuracy']
        linear_accuracy = svd_attack_data[idx]['linear']['accuracy']

        plt.figure(figsize=(10, 5))
        plt.plot(relu_accuracy.numpy(), label='ReLU Accuracy', marker='o')
        plt.plot(linear_accuracy.numpy(), label='Linear Accuracy', marker='x')
        plt.title(f'Model Accuracy vs SVD Components Ablated', fontsize=20)
        plt.xlabel('Top-k Components Removed', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)

        path = '/Users/alicerigg/Code/polyapprox/experiments/figs/new_plotting/svd12.pdf'
        if savefig:
            plt.savefig(path)
        plt.show()

if __name__ == "__main__":
    main()

