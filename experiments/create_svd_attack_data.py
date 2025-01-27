from experiments.plotting_functional import svd_attack_projection, evaluate_model, ckpt_to_model, startup
import torch




'''
Reproduces SVD attack data.
'''


if False:


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

    datadict['plotting_data']['svd_attack'] = svd_attack_data

    torch.save(datadict, datadict_path)