import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from mnist_2l import Minimal_FFN
from decomp.model import FFNModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Union, Literal, Callable
import numpy as np
from tqdm import tqdm

DIR = '/Users/alicerigg/Code/polyapprox/experiments/results/descent/'
savepaths = {}

def kl_divergence(px, qx):
    assert px.shape == qx.shape, ValueError(f'Shapes must match! {px.shape} != {qx.shape}')
    # expecting -1 dim to be class logits in probability form. softmax just in case
    p_x = torch.nn.functional.log_softmax(px, dim=-1)
    q_x = torch.nn.functional.log_softmax(qx, dim=-1)
    kl_divergence = torch.sum(torch.exp(p_x) * (p_x - q_x), dim=-1)
    return kl_divergence

class Plotter:
    def __init__(self, mlp_model, dataset):
        self.util = Utilizer(mlp_model)
        self.eval = Evaluator(dataset)
        self.orders = ['relu', 'linear', 'quadratic']

    def evaluate_kl_div(self, qmodel, **kwargs):
        p, _ = self.eval.evaluate(self.util.model, **kwargs)
        q, _ = self.eval.evaluate(qmodel, **kwargs)
        kl_div = kl_divergence(p,q).detach()
        return (kl_div.sum() / len(kl_div))
    
    def evaluate_fvu(self, qmodel, **kwargs):
        # Evaluate the logits for both models
        p_logits, _ = self.eval.evaluate(self.util.model, **kwargs)
        q_logits, _ = self.eval.evaluate(qmodel, **kwargs)
        
        # Calculate the variance of the pmodel logits
        total_variance = torch.var(p_logits, unbiased=False)
        
        # Calculate the residual variance (variance of the difference between pmodel and qmodel logits)
        residual_variance = torch.var(p_logits - q_logits, unbiased=False)
        
        # Calculate the fraction of variance unexplained (FVU)
        fvu = residual_variance / total_variance
        
        return fvu.item()
    
    def preview_dataset(self, num_images=10, figsize=(10,5), width=5):
        plt.figure(figsize=figsize)
        height = (num_images + width-1) // width
        shape = self.eval.get_input_shape()
        for i in range(num_images):
            plt.subplot(height, width, i+1)
            if self.eval.get_dataset_name() in ['svhn', 'cifar']:
                plt.imshow(self.eval.dataset.x[i].reshape(shape).permute(1, 2, 0), cmap=None)  # Remove cmap for RGB
            else:
                plt.imshow(self.eval.dataset.x[i].reshape(shape), cmap='gray')
            plt.title(f"Image {i+1}, Label: {self.eval.dataset.y[i].item()}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_spectrum(self, eigvals, description=''):
        plt.figure(figsize=(8, 6))
        plt.plot(eigvals, marker='o')
        plt.title(f'Eigenvalues for {description}')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        plt.show()
    
    def get_top_eigenvectors(self, out_dir: Union[torch.Tensor, int],
                             topk=3, orientation: Literal[None, 1, -1]=None):
        eigvals, eigvecs = self.util.get_eigdecomp(out_dir, orientation=orientation)
        shape = self.eval.get_input_shape()

        top_pos_eigvals, top_neg_eigvals = torch.zeros(2, topk)
        top_pos_eigvecs, top_neg_eigvecs = torch.zeros(2, topk, self.util.d_input)
        
        for i in range(topk):
            top_pos_eigvecs[i] = eigvecs[:,-(i+1)]
            top_neg_eigvecs[i] = eigvecs[:,i]
            top_pos_eigvals[i] = eigvals[-(i+1)]
            top_neg_eigvals[i] = eigvals[i]

        top_pos_eigvecs = top_pos_eigvecs.reshape(-1, *shape)
        top_neg_eigvecs = top_neg_eigvecs.reshape(-1, *shape)

        return top_pos_eigvals, top_neg_eigvals, top_pos_eigvecs, top_neg_eigvecs

    def plot_top_eigenvectors(self, out_dir, topk=3, vmax=0.2):
        pos_eigvals, neg_eigvals, pos_eigvecs, neg_eigvecs = self.get_top_eigenvectors(out_dir, topk)
        plt.figure(figsize=(12, 9))
        dataset_name = self.eval.get_dataset_name()
        # Determine vmin and vmax for consistent color scaling
        vmin = -vmax
        # Display top positive eigenvectors
        for i in range(topk):
            plt.subplot(2, topk, i+1)
            assert dataset_name in self.eval._supported_datasets(),\
                ValueError(f'Dataset {dataset_name} not supported!')
            plt.imshow(pos_eigvecs[i], cmap='RdBu', vmin=vmin, vmax=vmax)
            #img_pos = plt.imshow(eigvecs[:,-(i+1)].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.title(f'Top Positive Eigenvector {i+1}\nEigenvalue: {pos_eigvals[i]:.2f}')
            plt.axis('off')
        # Display top negative eigenvectors
        for i in range(topk):
            plt.subplot(2, topk, i+topk+1)
            plt.imshow(neg_eigvecs[i], cmap='RdBu', vmin=vmin, vmax=vmax)
            #img_neg = plt.imshow(eigvecs[:,i].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.title(f'Top Negative Eigenvector {i+1}\nEigenvalue: {neg_eigvals[i]:.2f}')
            plt.axis('off')
        # Add a colorbar to indicate the scale
        #plt.colorbar(img, ax=plt.gcf().axes, orientation='horizontal', fraction=0.05, pad=0.1)
        plt.tight_layout()
        plt.show()

    def return_top_eigenvector_fig(self, out_dir, vmax=0.2, show_beta=-1,
                                   pos_description='Top Positive Eigenvector 1\n',
                                   neg_description='Top Negative Eigenvector 1\n',
                                   val_description='Eigenvalue: ',
                                   orientation=None):
        # Get the eigenvalues and eigenvectors
        pos_eigvals, neg_eigvals, pos_eigvecs, neg_eigvecs = \
            self.get_top_eigenvectors(out_dir, topk=1, orientation=orientation)
        #dataset_name = self.eval.get_dataset_name()
        vmin = -vmax
        
        # Create a single figure with a 2 x topk layout
        if show_beta != -1:
            num_rows = 3
        else:
            num_rows = 2
        fig, axes = plt.subplots(num_rows, 1, figsize=(4, 6))

        axes = [[axes[i]] for i in range(num_rows)] 
        # Render the positive eigenvector
        ax = axes[0][0]
        ax.imshow(pos_eigvecs[0], cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'{pos_description}{val_description}{pos_eigvals[0]:.2f}')
        ax.axis('off')
    
        # Plot top negative eigenvector (second row)

        ax = axes[1][0]
        ax.imshow(neg_eigvecs[0], cmap='RdBu', vmin=vmin, vmax=vmax)
        ax.set_title(f'{neg_description}{val_description}{neg_eigvals[0]:.2f}')
        ax.axis('off')

        if show_beta != -1:
            shape = self.eval.get_input_shape()
            ax = axes[2][0]
            ax.imshow(self.util.ols.beta[:,show_beta].reshape(shape),
                      cmap='RdBu', vmin=vmin, vmax=vmax)
            #ax.set_title(f'{neg_description}{val_description}{neg_eigvals[0]:.2f}')
            #ax.set_title(f'{neg_description}{val_description}{neg_eigvals[0]:.2f}')
            ax.axis('off')

        fig.tight_layout()

        return fig, axes
    
    def plot_beta_rows(self, figsize=(10,5), vmax=0.2, show_cbar=False):
        # currently only working for mnist. Above stuff as well.
        # Need to try cifar separately and see what configs work for plt.imshow
        shape = self.eval.get_input_shape()
        dataset_name = self.eval.get_dataset_name()
        
        assert dataset_name in self.eval._supported_datasets(),\
                ValueError(f'Dataset {dataset_name} not supported!')
        plt.figure(figsize=figsize)
        vmin = -vmax

        for i, row in enumerate(self.util.ols.beta.T):
            plt.subplot(2, 5, i+1)
            img = plt.imshow(row.reshape(shape), cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.title(f'Beta row {i}')
            plt.axis('off')
        
        if show_cbar:
            cbar_ax = plt.gcf().add_axes([0.15, 0.47, 0.7, 0.02])  # [left, bottom, width, height]
            plt.colorbar(img, cax=cbar_ax, orientation='horizontal')
        
        plt.tight_layout()
        plt.show()

    def plot_svd_components(self, num_components=-1, figsize=(10,10), vmax=0.2):
        shape = self.eval.get_input_shape()
        dataset_name = self.eval.get_dataset_name()
        u, s, vh = self.util.get_svd()
        if num_components == -1 or num_components > len(vh[0]):
            num_components = len(vh[0])
        print(u.shape)
        plt.figure(figsize=(10, 10))
        #shape = (3,32,32)
        # Determine the global max for the color scale and set vmin to -vmax
        vmin = -vmax
        # Separate u into positive and negative terms
        #u_positive = torch.clamp(u, min=0) # unused
        #u_negative = -torch.clamp(u, max=0)

        # Plot the positive and negative components separately
        plt.figure(figsize=(15, 13))

        for i in range(num_components):
            # Plot positive components
            plt.subplot(4, 5, i+1)
            assert dataset_name in self.eval._supported_datasets(),\
                ValueError(f'Dataset {dataset_name} not supported!')
            #if dataset_name in ['svhn', 'cifar']:
            #    img = plt.imshow(u[:,i].reshape(shape).permute(1, 2, 0), cmap=None)
            #else:
            img = plt.imshow(u[:,i].reshape(shape), cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.title(f'Component {i+1}, value {s[i]:.2f}')
            plt.axis('off')

        # Add a colorbar to indicate the scale, placed in the middle and horizontal
        #cbar_ax = plt.gcf().add_axes([0.15, 0.47, 0.7, 0.02])
        #plt.colorbar(img, cax=cbar_ax, orientation='horizontal')

        plt.tight_layout()
        plt.show()
    
    def plot_stacked_logits(self, image, figsize=(10,5), compare_to_base=False, barwidth=0.35):
        zeroth, first, second = self.util.ols(image, split_logits=True)
        zeroth, first, second = zeroth.squeeze(), first.squeeze(), second.squeeze()
        
        indices = range(len(zeroth))
        
        fig, ax = plt.subplots(figsize=figsize)
        # Separate positive and negative parts for each component
        pos0, neg0 = zeroth.clip(min=0), zeroth.clip(max=0)
        pos1, neg1 = first.clip(min=0), first.clip(max=0)
        pos2, neg2 = second.clip(min=0), second.clip(max=0)
        
        if compare_to_base:
            real_logits = self.util.model(image).squeeze()
            indices = [i + barwidth/2 for i in range(len(zeroth))]
            real_indices = [i - barwidth/2 for i in range(len(zeroth))]
            plt.bar(real_indices, real_logits, width=barwidth,
                    color='purple', alpha=0.7, label='ReLU logits')
        # Plot stacked bar chart for positive values
        ax.bar(indices, pos2, width=barwidth, color='blue', label='Gamma Logits (Positive)')
        ax.bar(indices, pos1, width=barwidth, bottom=pos2, color='orange', label='Beta Logits (Positive)')
        ax.bar(indices, pos0, width=barwidth, bottom=pos2 + pos1, color='green', label='Alpha Logits (Positive)')

        # Plot stacked bar chart for negative values
        ax.bar(indices, neg2, width=barwidth, color='blue', label='Gamma Logits (Negative)')
        ax.bar(indices, neg1, width=barwidth, bottom=neg2, color='orange', label='Beta Logits (Negative)')
        ax.bar(indices, neg0, width=barwidth, bottom=neg2 + neg1, color='green', label='Alpha Logits (Negative)')

        ax.set_title('Stacked Logits')
        ax.set_xlabel('Logit Index')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def get_svd_attack_data(self):
        data = {'acc': torch.zeros(3,11),
                'kl': torch.zeros(3,11),
                'fvu': torch.zeros(3,11)}
        test_models = [self.util.model,
                       self.util.ols.to_linear(),
                       self.util.ols]
        
        for i in range(11):
            proj = self.util.svd_attack_projection(topk=i)
            for j, model in enumerate(test_models):
                data['acc'][j,i] = self.eval.evaluate(
                    model, proj=proj, return_logits=False)
                if j >= 1:
                    data['kl'][j][i] = self.evaluate_kl_div(
                    model, proj=proj, return_logits=True)
                    data['fvu'][j][i] = self.evaluate_fvu(
                    model, proj=proj, return_logits=True)
        return data
    def _configure_axis(self, ax, title, xlabel, ylabel, xlogscale, ylogscale, ymin, ymax):
        """Helper function to configure the common settings for a plot axis."""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlogscale:
            ax.set_xscale('log')  # Set x-axis to log scale
        if ylogscale:
            ax.set_yscale('log')  # Set y-axis to log scale
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.grid(True)
        ax.legend()

    def _plot_measure(self, ax, data, measure, start_i):
        """Helper function to plot data for a specific measure."""
        for i in range(start_i, 3):
            ax.plot(range(11), data[measure][i, :],
                    label=f"{self.orders[i]} {measure}",
                    color=cm.viridis(float(i) * 0.2 + 0.4))

    def plot_generic(self, data, x_range=None,
                    xlogscale=False, ylogscale=False,
                    ymin=None, ymax=None):
        """
        Generic plot method for 2D data.

        Parameters:
            data (Tensor): 2D tensor where the first dimension is the number of traces and the second is the data to be plotted.
            x_range (iterable, optional): Custom x-axis range. Defaults to range(data.size(1)).
            xlogscale (bool, optional): Whether to use a logarithmic scale for the x-axis.
            ylogscale (bool, optional): Whether to use a logarithmic scale for the y-axis.
            ymin (float, optional): Minimum y-axis limit.
            ymax (float, optional): Maximum y-axis limit.
        """
        if x_range is None:
            x_range = range(data.size(1))

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(data.size(0)):
            ax.plot(x_range, data[i, :], label=f"Trace {i+1}", color=cm.viridis(float(i) / data.size(0)))

        self._configure_axis(ax, "Generic Plot", "X-axis", "Y-axis", xlogscale, ylogscale, ymin, ymax)
        plt.tight_layout()
        plt.show()

    def _plot_eig_key(self, ax, data, key, x_axis):
        """Helper function to plot data for a specific measure."""
        for i in range(data[key].size(0)):
            for j in range(data[key].size(1)):
                ax.plot(x_axis, data[key][i, :],
                        label=f"{self.orders[i]} {key}",
                        color=cm.viridis(float(i) * 0.2 + 0.4))
        pass
    def _plot_svd_attack(self, data, mode=['acc'],
                        xlogscale=False, ylogscale=False,
                        ymin=None, ymax=None, multiplot=False):
        if 'all' in mode:
            mode = ['acc', 'kl', 'fvu']

        if multiplot:
            # Create subplots for multiple measures
            fig, axes = plt.subplots(1, len(mode), figsize=(10 * len(mode), 6))
            if len(mode) == 1:  # Handle the case where there's only one mode
                axes = [axes]

            for idx, measure in enumerate(mode):
                ax = axes[idx]
                start_i = 0 if measure == 'acc' else 1
                self._plot_measure(ax, data, measure, start_i)
                self._configure_axis(ax, f'{measure.upper()} vs SVD components Ablated',
                                     'SVD Components Ablated', f'{measure.upper()}',
                                     xlogscale, ylogscale, ymin, ymax)
            plt.tight_layout()
            plt.show()
        else:
            # Separate plots for each measure
            for measure in mode:
                fig, ax = plt.subplots(figsize=(10, 6))
                start_i = 0 if measure == 'acc' else 1
                self._plot_measure(ax, data, measure, start_i)
                self._configure_axis(ax, f'{measure.upper()} vs SVD components Ablated',
                                     'SVD Components Ablated', f'{measure.upper()}',
                                     xlogscale, ylogscale, ymin, ymax)
                plt.show()    
    def plot_svd_attack(self, data, mode=['acc'],
                        xlogscale=True, ylogscale=False,
                        ymin=None, ymax=None, multiplot=False):
        orders = ['relu', 'linear', 'quadratic']
        if 'all' in mode:
            mode = ['acc', 'kl', 'fvu']

        if multiplot:
            fig, axes = plt.subplots(1, len(mode), figsize=(10 * len(mode), 6))
            for idx, measure in enumerate(mode):
                ax = axes[idx]
                start_i = 0
                if measure != 'acc':
                    start_i = 1
                for i in range(start_i, 3):
                    ax.plot(range(11), data[measure][i,:],
                            label=f"{orders[i]} {measure}",
                            color=cm.viridis(float(i)*0.2 + 0.4))
                
                ax.set_xlabel('Steps')
                ax.set_ylabel(f'{measure.upper()}')
                ax.set_title(f'{measure.upper()} vs SVD components Ablated')

                if xlogscale:
                    ax.set_xscale('log')  # Set x-axis to log scale
                if ylogscale:
                    ax.set_yscale('log')  # Set y-axis to log scale

                ax.set_ylim(ymin=ymin, ymax=ymax)
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            for measure in mode:
                plt.figure(figsize=(10, 6))
                start_i = 0
                if measure != 'acc':
                    start_i = 1
                for i in range(start_i, 3):
                    plt.plot(range(11), data[measure][i,:],
                             label=f"{orders[i]} {measure}",
                             color=cm.viridis(float(i)*0.2 + 0.4))
                
                plt.xlabel('Steps')
                plt.ylabel(f'{measure.upper()}')
                plt.title(f'{measure.upper()} vs SVD components Ablated')

                if xlogscale:
                    plt.xscale('log')  # Set x-axis to log scale
                if ylogscale:
                    plt.yscale('log')  # Set y-axis to log scale

                plt.ylim(ymin=ymin, ymax=ymax)
                plt.legend()
                plt.grid(True)
                plt.show()
    
    def get_eig_attack_data(self, out_dir=3, topk=1, std = torch.linspace(0,1,20)):
        # shape (3,2,n)
        # 3: relu, linear, quadratic
        # 2: attack, noise baseline
        # n: std sweep
        input_shape = self.eval.dataset.x.flatten(start_dim=1).shape
        data = {'acc': torch.zeros(3,3,std.size(0)),
                'kl': torch.zeros(3,3,std.size(0)),
                'fvu': torch.zeros(3,3,std.size(0))}
        test_models = [self.util.model,
                       self.util.ols.to_linear(),
                       self.util.ols]
        
        data['attack'] = []
        for i in tqdm(range(std.size(0))):
            add = self.util.eig_attack_projection(out_dir=out_dir,topk=topk, std=std[i])
            print(f'add norm: {torch.norm(add)}, vs std: {std[i]}')
            #add_noise_std = std[i]
            #interventions = [add, add_noise, None]
            for j, model in enumerate(test_models):
            #for k, intervention in enumerate(interventions):

                data['acc'][j,0,i] = self.eval.evaluate(
                    model, add=add, return_logits=False)
                data['acc'][j,1,i] = self.eval.evaluate(
                    model, add_noise_std=std[i], return_logits=False)
                data['acc'][j,2,i] = self.eval.evaluate(
                    model, return_logits=False)
                #data['acc'][j,1,i] = self.eval.evaluate(
                #    model, add_noise_std=noise, return_logits=False)
                if j >= 1:
                    data['kl'][j,0,i] = self.evaluate_kl_div(
                    model, add=add, return_logits=True)
                    data['kl'][j,1,i] = self.evaluate_kl_div(
                    model, add_noise_std=std[i], return_logits=True)
                    data['kl'][j,2,i] = self.evaluate_kl_div(
                    model, return_logits=True)

                    data['fvu'][j,0,i] = self.evaluate_fvu(
                    model, add=add, return_logits=True)
                    data['fvu'][j,1,i] = self.evaluate_fvu(
                    model, add_noise_std=std[i], return_logits=True)
                    data['fvu'][j,2,i] = self.evaluate_fvu(
                    model, return_logits=True)
            data['attack'].append(add)
                    #    data['kl'][j,0,i] = self.evaluate_kl_div(
                    #    model, add=add, return_logits=True)
                    #    data['fvu'][j,0,i] = self.evaluate_fvu(
                    #    model, add=add, return_logits=True)
                    #    data['kl'][j,1,i] = self.evaluate_kl_div(
                    #    model, add_noise_std=noise, return_logits=True)
                    #    data['fvu'][j,1,i] = self.evaluate_fvu(
                    #    model, add_noise_std=noise, return_logits=True)
        return data
    
    def get_eig_attack_data_all(self, topk=1, std = torch.linspace(0,1,20)):
        data = {'acc': torch.zeros(3,10,std.size(0)),
                'kl': torch.zeros(3,10,std.size(0)),
                'fvu': torch.zeros(3,10,std.size(0))}
        test_models = [self.util.model,
                       self.util.ols.to_linear(),
                       self.util.ols]
        
        for i in tqdm(range(std.size(0))):
            for digit in range(10):
                add = self.util.eig_attack_projection(out_dir=digit,topk=topk, std=std[i])
                for j, model in enumerate(test_models):
                    data['acc'][j,digit,i] = self.eval.evaluate(
                        model, add=add, return_logits=False)
                    if j >= 1:
                        data['kl'][j,digit,i] = self.evaluate_kl_div(
                        model, add=add, return_logits=True)
                        data['fvu'][j,digit,i] = self.evaluate_fvu(
                        model, add=add, return_logits=True)
        return data

    def _plot_eig_attack(self, data, mode=['acc'],
                        xlogscale=False, ylogscale=False,
                        ymin=None, ymax=None, multiplot=False):
        """
        Plots eigenvalue attack results for specified measures.
        """
        if 'all' in mode:
            mode = ['acc', 'kl', 'fvu']

        n = data['acc'].shape[-1]  # Get the size of the last dimension
        print(n)
        if multiplot:
            # Create subplots for multiple measures
            fig, axes = plt.subplots(1, len(mode), figsize=(10 * len(mode), 6))
            if len(mode) == 1:  # Handle the case where mode has only one element
                axes = [axes]

            for idx, measure in enumerate(mode):
                ax = axes[idx]
                start_i = 0 if measure == 'acc' else 1
                self._plot_measure(ax, data, measure, start_i)  # Use helper function to plot
                self._configure_axis(ax, f'{measure.upper()} vs Attack Magnitude',
                                    'Steps', f'{measure.upper()}',
                                    xlogscale, ylogscale, ymin, ymax)  # Axis configuration
            plt.tight_layout()
            plt.show()
        else:
            # Create a separate plot for each measure
            for measure in mode:
                fig, ax = plt.subplots(figsize=(10, 6))
                start_i = 0 if measure == 'acc' else 1
                self._plot_measure(ax, data, measure, start_i)  # Use helper function to plot
                self._configure_axis(ax, f'{measure.upper()} vs Attack Magnitude',
                                    'Steps', f'{measure.upper()}',
                                    xlogscale, ylogscale, ymin, ymax)  # Axis configuration
                plt.show()

    def plot_eig_attack(self, data, mode=['acc'],
                        xlogscale=False, ylogscale=False,
                        ymin=None, ymax=None, multiplot=False):
        '''

        '''
        orders = ['relu', 'linear', 'quadratic']
        if 'all' in mode:
            mode = ['acc', 'kl', 'fvu']
        n = data['acc'].size(-1)

        if multiplot:
            fig, axes = plt.subplots(1, len(mode), figsize=(10 * len(mode), 6))
            for idx, measure in enumerate(mode):
                ax = axes[idx]
                start_i = 0
                if measure != 'acc':
                    start_i = 1
                for i in range(start_i, 3):
                    ax.plot(range(n), data[measure][i,:],
                            label=f"{orders[i]} {measure}",
                            color=cm.viridis(float(i)*0.2 + 0.4))
                
                ax.set_xlabel('Steps')
                ax.set_ylabel(f'{measure.upper()}')
                ax.set_title(f'{measure.upper()} vs Attack Magnitude')

                if xlogscale:
                    ax.set_xscale('log')  # Set x-axis to log scale
                if ylogscale:
                    ax.set_yscale('log')  # Set y-axis to log scale

                ax.set_ylim(ymin=ymin, ymax=ymax)
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            for measure in mode:
                plt.figure(figsize=(10, 6))
                start_i = 0
                if measure != 'acc':
                    start_i = 1
                for i in range(start_i, 3):
                    plt.plot(range(n), data[measure][i,:],
                             label=f"{orders[i]} {measure}",
                             color=cm.viridis(float(i)*0.2 + 0.4))
                
                plt.xlabel('Steps')
                plt.ylabel(f'{measure.upper()}')
                plt.title(f'{measure.upper()} vs Attack Magnitude')

                if xlogscale:
                    plt.xscale('log')  # Set x-axis to log scale
                if ylogscale:
                    plt.yscale('log')  # Set y-axis to log scale

                plt.ylim(ymin=ymin, ymax=ymax)
                plt.legend()
                plt.grid(True)
                plt.show()


class MultiPlotter:
    def __init__(self,
                 plotter_list: Union[list[Plotter], None] = None,
                 measures=None):
        self.plotters = plotter_list
        self.measures = measures
        self.n = len(plotter_list)

    def _init_plotters(self, plotter_list: list[Plotter]):
        assert isinstance(plotter_list, list) and all(isinstance(p, Plotter) for p in plotter_list) ;
        "plotter_list must be a list of Plotter instances"
        self.plotters = plotter_list

    def evaluate_quantitative_measures(self, num_samples=10000):


        m = self.plotters[-1]
        data, measures = {}, {'linear': {}, 'quadratic': {}, 'relu': {}}
        data['test'] = m.eval.dataset.x[:num_samples]
        data['new'] = m.eval.psd_generator.sample((num_samples,))
        #data['psd'] = eval.psd_generator.sample((num_samples,))
        data['acc'] = None
        #data['loss'] = None
        data['n01'] = m.eval.psd_generator.base_generator.sample((num_samples,))

        # populate dict
        for key in data.keys():
            # simply do not use extra dimensions. Bad practice, but who cares for now.
            measures['relu'][key] = torch.zeros(self.n,2)
            measures['linear'][key] = torch.zeros(self.n,2)
            measures['quadratic'][key] = torch.zeros(self.n,2)
        for i, ckpt in enumerate(self.plotters):
            #measures['weight_norm'][i] = util.get_weight_norm()
            for key, val in data.items():
                if key == 'acc':
                    measures['relu'][key][i,0] = m.eval.evaluate(ckpt.util.model, return_logits=False)
                    measures['linear'][key][i,0] = m.eval.evaluate(ckpt.util.ols.to_linear(), return_logits=False)
                    measures['quadratic'][key][i,0] = m.eval.evaluate(ckpt.util.ols, return_logits=False)
                else:
                    # relative measures: zero init for 'relu' is correct.
                    measures['linear'][key][i,0] = ckpt.evaluate_kl_div(ckpt.util.ols.to_linear(), inputs=val)
                    measures['linear'][key][i,1] = ckpt.evaluate_fvu(ckpt.util.ols.to_linear(), inputs=val)
                    measures['quadratic'][key][i,0] = ckpt.evaluate_kl_div(ckpt.util.ols, inputs=val)
                    measures['quadratic'][key][i,1] = ckpt.evaluate_fvu(ckpt.util.ols, inputs=val)
        
        self.measures = measures
        return measures
   
    def _configure_axis(self, ax, title, xlabel, ylabel, xlogscale, ylogscale, ymin, ymax):
        """Helper function to configure the axis."""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlogscale:
            ax.set_xscale("log")
        if ylogscale:
            ax.set_yscale("log")
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.grid(True)
        ax.legend()

    def _plot_measure(self, ax, data, steps, linestyles, colors, measure_type, y_index):
        """Helper function to plot specific measures."""
        for order, measure_dict in data.items():
            if measure_type in measure_dict:
                y_data = measure_dict[measure_type][:, y_index]
                ax.plot(
                    steps,
                    y_data,
                    label=f"{order} {measure_type}",
                    linestyle=linestyles[order],
                    color=colors.get(measure_type, "blue"),
                )
                    
    def plot_accuracies(self, savepath=None, dir='acc',
                        xlogscale=True, ylogscale=False,
                        ymin=None, ymax=None, format='pdf', custom_params=None):
        if self.measures is None:
            raise ValueError('Measures not set yet. Run evaluate_quantitative_measures first.')

        default_params = {
        'font.size': 12,                  # General font size
        'axes.titlesize': 16,            # Title font size
        'axes.labelsize': 14,            # Axis label size
        'axes.titlepad': 20,             # Padding for the title
        'axes.labelpad': 10,             # Padding for axis labels
        'legend.fontsize': 12,           # Legend font size
        'xtick.labelsize': 12,           # X-axis tick label size
        'ytick.labelsize': 12,           # Y-axis tick label size
        'grid.alpha': 0.7,               # Grid transparency
        'grid.linestyle': '--',          # Grid line style
        'grid.linewidth': 0.5,           # Grid line width
        'title': f'Accuracy over Steps for {savepath}',  # Default title
        'title_fontsize': None,          # Default font size for the title
        }
        custom_params = custom_params or {}
        merged_params = {**default_params, **custom_params}
        
        with plt.rc_context(rc=custom_params):
            plt.figure(figsize=(10, 6))
            steps = [2**i for i in range(self.measures['linear']['test'].size(0))]
            linestyles = {'quadratic': '--', 'linear': ':', 'relu': '-'}
            
            colors = {'acc': cm.viridis}
            scale = np.linspace(0.9, 0.4, len(self.measures))
            if savepath is not None:
                savepath = savepaths.get(savepath, savepath)

            for i, (order, acc_dict) in enumerate(self.measures.items()):
                plt.plot(steps, acc_dict['acc'][:,0],
                        label=f"{order} accuracy",
                        linestyle=linestyles[order],
                        color=colors['acc'](scale[i]))
            
            plt.xlabel('Steps')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy over Steps for {savepath}')

            if xlogscale:
                plt.xscale('log')  # Set x-axis to log scale
            if ylogscale:
                plt.yscale('log')  # Set y-axis to log scale
            
            plt.ylim(ymin=ymin, ymax=ymax)
            plt.legend()
            plt.grid(True)
            if savepath is not None:
                plt.savefig(f'{DIR}{savepath}/{dir}.{format}', format=format)

            plt.show()
    
    def plot_fvu(self, savepath=None, dir='fvu',
                 xlogscale=True, ylogscale=False,
                 ymin=None, ymax=None, format='pdf'):
        if self.measures is None:
            raise ValueError('Measures not set yet. Run evaluate_quantitative_measures first.')
        plt.figure(figsize=(10, 6))
        steps = [2**i for i in range(self.measures['linear']['test'].size(0))]
        linestyles = {'quadratic': '--', 'linear': ':', 'relu': '-'}
        colors = {'test': 'red', 'new': 'orange', 'n01': 'green', 'acc': 'blue'}
        if savepath is not None:
            savepath = savepaths.get(savepath, savepath)

        
        for order, fvu_dict in self.measures.items():
            if order != 'relu':
                for key in fvu_dict.keys() - {'acc'}:
                    plt.plot(steps, fvu_dict[key][:,1],
                                label=f"{order} {key}",
                                linestyle=linestyles[order],
                                color=colors[key])

        plt.xlabel('Steps')
        plt.ylabel('Fraction of Variance Unexplained (FVU)')
        plt.title(f'FVU over Steps for {savepath}')

        if xlogscale:
            plt.xscale('log')  # Set x-axis to log scale
        if ylogscale:
            plt.yscale('log')  # Set y-axis to log scale
        
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.legend()
        plt.grid(True)
        if savepath is not None:
            plt.savefig(f'{DIR}{savepath}/{dir}.{format}', format=format)
        else:
            plt.show()

    def plot_kl_divergence(self, savepath=None, dir='kldiv',
                           xlogscale=True, ylogscale=False,
                           ymin=None, ymax=None, format='pdf'):
        if self.measures is None:
            raise ValueError('Measures not set yet. Run evaluate_quantitative_measures first.')
        plt.figure(figsize=(10, 6))
        steps = [2**i for i in range(self.measures['linear']['test'].size(0))]
        linestyles = {'quadratic': '--', 'linear': ':', 'relu': '-'}
        colors = {'test': 'red', 'new': 'orange', 'n01': 'green', 'acc': 'blue'}
        #linestyles = {}
        kldiv_linestyle='-'
        if savepath is not None:
            savepath = savepaths.get(savepath, savepath)

        
        for order, fvu_dict in self.measures.items():
            if order != 'relu':
                for key in fvu_dict.keys() - {'acc'}:
                    plt.plot(steps, fvu_dict[key][:,0],
                                label=f"{order} {key}",
                                #marker=markers[order],

                                color=colors[key],
                                linestyle=linestyles[order])

        plt.xlabel('Steps')
        plt.ylabel('KL Divergence')
        plt.title(f'KL Divergence over Steps for {savepath}')

        if xlogscale:
            plt.xscale('log')  # Set x-axis to log scale
        if ylogscale:
            plt.yscale('log')  # Set y-axis to log scale

        plt.ylim(ymin=ymin, ymax=ymax)
        plt.legend()
        plt.grid(True)
        if savepath is not None:
            plt.savefig(f'{DIR}{savepath}/{dir}.{format}', format=format)
        else:
            plt.show()
    
    def plot_eigvecs_over_list(self, subplots_adjust = None, show_beta=-1, savepath=None,
                               out_dir=3, vmax=0.2, col_start=0, col_end=None, figsize=(15,6),
                               text_pos=None, format='pdf', orientation=None):
        if col_end is None:
            num_cols = len(self.plotters) - col_start
        fig_composite, axes_composite = plt.subplots(nrows=2, ncols=num_cols, figsize=figsize)
        for i, ckpt in enumerate(self.plotters[col_start:col_end]):
            fig, axes = ckpt.return_top_eigenvector_fig(out_dir=out_dir, vmax=vmax,
                                                        show_beta=show_beta,
                                                        pos_description=f"Step $2^{{{col_start+i}}}$\n",
                                                        neg_description='',
                                                        val_description=r'$\lambda=$',
                                                        orientation=orientation)
            
            # Move positive eigenvector plot into the composite figure
            image_pos = axes[0][0].images[0].get_array()
            axes_composite[0, i].imshow(image_pos, cmap='RdBu', vmin=-0.2, vmax=0.2)
            axes_composite[0, i].set_title(axes[0][0].get_title())
            axes_composite[0, i].axis('off')

            if show_beta != -1:
                # Move beta plot into the composite figure
                image_beta = axes[2][0].images[0].get_array()
                axes_composite[1, i].imshow(image_beta, cmap='RdBu', vmin=-0.2, vmax=0.2)
                axes_composite[1, i].set_title(axes[2][0].get_title())
                axes_composite[1, i].axis('off')            

            plt.close(fig)

        if text_pos is None:
            text_pos = {'text_x': 0.00, 'text_y1': 0.67, 'text_y2': 0.28}

        if subplots_adjust is None:
            subplots_adjust = {'left': 0.2, 'right': 0.95, 'top': 0.95, 'bottom': 0.1}

        fig_composite.text(text_pos['text_x'], text_pos['text_y1'], "Quadratic", fontsize=16,
                            fontname='Times New Roman', fontweight='heavy',
                            rotation=90, va='center', ha='center')
        
        if show_beta != -1:
            fig_composite.text(text_pos['text_x'], text_pos['text_y2'], "Linear", fontsize=16,
                            fontname='Times New Roman', fontweight='heavy',
                            rotation=90, va='center', ha='center')
        # Adjust layout and display the composite plot

        #fig_composite.tight_layout()
        # rather than tight layout, manually adjust:
        fig_composite.subplots_adjust(**subplots_adjust)  # Increase left margin
        fig_composite.savefig(f'{DIR}{savepath}/eigvec{out_dir}.{format}',format=format)
        



    def unified_plot(
        self,
        measure_type: Literal['test', 'new', 'acc', 'n01', 'distributions'],
        ylabel,
        y_index=0,
        title=None,
        colormap=cm.viridis,
        linestyles=None,
        colors={'anything': 'red'},
        savepath=None,
        dir="plot",
        xlogscale=True,
        ylogscale=False,
        ymin=None,
        ymax=None,
        format='pdf',
    ):
        """
        Generalized plotting function to handle accuracy, FVU, KL divergence, etc.

        Parameters:
        - measure_type: The specific measure to plot (e.g., 'acc', 'fvu', 'kldiv').
        - ylabel: Label for the y-axis.
        - y_index: Index in the measure array to extract data (0 for KL divergence, 1 for FVU, etc.).
        - title: Custom title for the plot. If None, defaults to measure_type.
        - colormap: Colormap for the lines (default: viridis).
        - linestyles: Dictionary mapping orders (e.g., linear, quadratic, relu) to line styles.
        - colors: Dictionary mapping measure keys (e.g., 'test', 'acc') to specific colors.
        - savepath: If provided, saves the plot to a directory instead of displaying.
        - dir: Directory name for saving the plot (default: "plot").
        - xlogscale: Whether to use a logarithmic scale for the x-axis.
        - ylogscale: Whether to use a logarithmic scale for the y-axis.
        - ymin: Minimum y-axis limit.
        - ymax: Maximum y-axis limit.
        """
        if measure_type == 'distributions':
            vals = ['test', 'new', 'n01']
        else:
            vals = [measure_type]
        if self.measures is None:
            raise ValueError("Measures not set yet. Run evaluate_quantitative_measures first.")

        steps = [2**i for i in range(self.measures['linear']['test'].size(0))]
        linestyles = linestyles or {'quadratic': '--', 'linear': ':', 'relu': '-'}
        colors = colors or {'test': 'red', 'new': 'orange', 'n01': 'green', 'acc': cm.viridis}

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the measures
        for order, measure_dict in self.measures.items():
            if measure_type in measure_dict:
                for key, measure_data in measure_dict.items():
                    if measure_type in key:  # Plot only relevant data
                        ax.plot(
                            steps,
                            measure_data[:, y_index],
                            label=f"{order} {key}",
                            linestyle=linestyles.get(order, '-'),
                            color=colors.get(key, colormap(0.5)),
                        )

        # Configure the axis
        self._configure_axis(
            ax,
            title or f"{measure_type.upper()} over Steps",
            "Steps",
            ylabel,
            xlogscale,
            ylogscale,
            ymin,
            ymax,
        )

        # If a savepath is provided, save the plot; otherwise, show it
        if savepath is not None:
            plt.savefig(f"{DIR}{savepath}/{dir}.{format}", format=format)
        else:
            plt.show()

class Utilizer:
    def __init__(self, mlp_model):
        if isinstance(mlp_model, FFNModel):
            self.model = Minimal_FFN(mlp_model.get_layer_data())
        elif isinstance(mlp_model, Minimal_FFN):
            self.model = mlp_model
        else:
            raise ValueError(f'Expecting type FFNModel or Minimal_FFN, got {type(mlp_model)}')
        
        self.d_input = len(self.model.W1.detach()[0])
        self.d_hidden = len(self.model.b1.detach())
        self.d_output = len(self.model.b2.detach())
        self.pinv = None
        self.pinv_k = 0
        self.ols = self.model.approx_fit('quadratic')
        self.linear = self.ols.to_linear()

    def get_svd(self, vals_only=False):
        if vals_only:
            return torch.linalg.svdvals(self.ols.beta)
        else:
            return torch.linalg.svd(self.ols.beta)

    def get_weight_norm(self):
        """
        Calculates full weight norm of the model
        """
        return torch.norm(torch.nn.utils.parameters_to_vector(self.model.parameters())).item()
    
    def integer_to_ei(self, idx: int, length: int = 10):
        vec = torch.zeros(length)
        vec[idx] = 1
        return vec
    
    def format_out_dir(self, out_dir, length=10):
        if isinstance(out_dir, int):
            out_dir = self.integer_to_ei(out_dir, length)
        try:
            if isinstance(out_dir, torch.Tensor) and out_dir.dim() > 1:
                out_dir = out_dir.view(-1)
            assert out_dir.size(0) == length, ValueError(f'Expected out_dir of length {length}, got {out_dir.size(0)}')
            return out_dir
        except Exception as e:
            raise ValueError(f"Error reshaping out_dir: {e}")
        
    def orient_eigenvectors(self, eigvecs, orientation=None):
        if orientation is None:
            return eigvecs
        else:
            clamped_up_norms = torch.clamp(eigvecs, min=0).norm(dim=0)
            clamped_down_norms = torch.clamp(eigvecs, max=0).norm(dim=0)

            # indices (vecs) where positive L2 mass >= negative L2 mass
            pos_mask = clamped_up_norms >= clamped_down_norms

            # otherwise
            neg_mask = clamped_up_norms < clamped_down_norms

            if orientation == 1: # flip neg_mask to make all vecs positive mass dominant
                eigvecs[:,neg_mask] *= -1
            elif orientation == -1: # flip pos_mask to make all vecs negative mass dominant
                eigvecs[:,pos_mask] *= -1
            else:
                ValueError(f'Expecting provided orientation +-1, got {orientation}')
            
            return eigvecs

    def get_eigdecomp(self, out_dir: Union[torch.Tensor, int],
                      vals_only=False, orientation=None):
        out_dir = self.format_out_dir(out_dir)
        b = self.ols.get_gamma_tensor()

        interaction_matrix = torch.einsum('h,hij->ij', out_dir, self.ols.get_gamma_tensor())
        if self.d_input > 2948: # eigh breaks
            eigvals, eigvecs = torch.linalg.eig(interaction_matrix)
            eigvals, eigvecs = eigvals.real, eigvecs.real
            sorted_indices = torch.argsort(eigvals)
            eigvals = eigvals[sorted_indices]
            eigvecs = eigvecs[:, sorted_indices]
        else:
            eigvals, eigvecs = torch.linalg.eigh(interaction_matrix)

        if orientation is not None:
            eigvecs = self.orient_eigenvectors(eigvecs, orientation)
        
        if vals_only:
            return eigvals
        else:
            return eigvals, eigvecs

    def get_top_eigenvectors(self, out_dir: Union[torch.Tensor, int],
                             topk=3, orientation=None):
        eigvals, eigvecs = self.get_eigdecomp(out_dir, orientation=orientation)
        #shape = self.eval.get_input_shape()

        top_pos_eigvals, top_neg_eigvals = torch.zeros(2, topk)
        top_pos_eigvecs, top_neg_eigvecs = torch.zeros(2, topk, self.d_input)

        for i in range(topk):
            top_pos_eigvecs[i] = eigvecs[:,-(i+1)]
            top_neg_eigvecs[i] = eigvecs[:,i]
            top_pos_eigvals[i] = eigvals[-(i+1)]
            top_neg_eigvals[i] = eigvals[i]
        #top_pos_eigvecs = top_pos_eigvecs.reshape(-1, *shape)
        #top_neg_eigvecs = top_neg_eigvecs.reshape(-1, *shape)

        return top_pos_eigvals, top_neg_eigvals, top_pos_eigvecs, top_neg_eigvecs

    def get_full_eigendecomp(self):
        if len(self.get_input_shape()) > 2948: # eigh breaks
            eigvals, eigvecs = torch.linalg.eig(self.ols.get_gamma_tensor())
            eigvals, eigvecs = eigvals.real, eigvecs.real
            sorted_indices = eigvals.argsort()
            return eigvals[sorted_indices], eigvecs[:, sorted_indices]
        else:
            return torch.linalg.eigh(self.ols.get_gamma_tensor())
    
    def svd_attack_projection(self, topk=1):
        u_mat, _, _ = torch.svd(self.ols.beta)
        P = torch.eye(u_mat.shape[0])
        for i in range(topk):
            P -= torch.outer(u_mat[:,i], u_mat[:,i])
        return P
    
    def get_full_eig_attack_matrix(self, topk=1, return_vecs=False):
        if self.pinv is not None:
            if topk <= self.pinv_k:
                print('Bigger pinv already computed, returning existing pinv')
                return self.pinv

        all_top_eigvals, all_top_eigvecs = torch.zeros(10, topk), torch.zeros(10, topk, self.d_input)
        for i in range(10):
            pos_vals, neg_vals, pos_vecs, neg_vecs = self.get_top_eigenvectors(i, topk=topk)
            all_top_eigvals[i], all_top_eigvecs[i] = pos_vals, pos_vecs
        
        # not necessary to project same-class components:
        # they are orthogonal already, will be a no-op. Redundant but simpler code

        all_top_eigvecs = all_top_eigvecs.view(10 * topk, -1)

        pseudoinverse = torch.linalg.pinv(all_top_eigvecs)
        # shape d_input by 10k. slice with [:,i*k] for class i
        self.pinv = pseudoinverse
        self.pinv_k = topk
        if return_vecs:
            return all_top_eigvecs, pseudoinverse
        else:
            return pseudoinverse
        
    def eig_attack_projection(self, out_dir: int, topk=1, std=1.0):
        # only need top positive eigvecs here
        if self.pinv_k >= topk:
            pseudoinverse = torch.zeros(self.d_input, 10*topk)
            for i in range(10):
                pseudoinverse[:,i*topk:(i+1)*topk] = self.pinv[:,i*self.pinv_k:i*self.pinv_k+topk]
        else:
            pseudoinverse = self.get_full_eig_attack_matrix(topk=topk)
        
        # shape 784, 30 (for topk=3)
        print(f'Current attack norm: {torch.norm(pseudoinverse[:,out_dir*topk])}')
        return std * pseudoinverse[:,out_dir*topk]
        #print(top_eigvecs.shape)


class GaussianMixture:
    def __init__(
        self,
        means: Tensor,
        covs: Tensor,
        class_probs: Tensor,
        size: int,
        shape: tuple[int, int, int] = (3, 32, 32),
        trf: Callable = lambda x: x, # transform
    ):
        self.class_probs = class_probs
        self.dists = [MultivariateNormal(mean, cov) for mean, cov in zip(means, covs)]
        self.shape = shape
        self.size = size
        self.trf = trf

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        y = torch.multinomial(self.class_probs, 1).squeeze()
        x = self.dists[y].sample().reshape(self.shape)
        return {
            "pixel_values": self.trf(x),
            "label": y,
        }

    def __len__(self) -> int:
        return self.size
    
class PSD_Generator:
    def __init__(self, mean, psd_sqrt_matrix):
        self.d_input = mean.size(0)
        self.mean = mean
        self.psd_sqrt_matrix = psd_sqrt_matrix
        self.base_generator = MultivariateNormal(
            torch.zeros(self.d_input),
            torch.eye(self.d_input)
        )

    def sample(self, sample_shape: torch.Size):
        data = self.base_generator.sample(sample_shape) # expected: (*,d_input)
        return data @ self.psd_sqrt_matrix + self.mean[None, :]
    
class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_size = len(dataset.y.flatten())
        self.generator = None
        self.psd_generator = None
        self.acc_loss = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
          
    def init_generator(self, use_psd_sqrt=False):
        '''
        computes mean, cov of entire self.dataset (subset not yet supported)
        Usage:
        self.generator.sample((5,10,25))
        >>> returns a tensor of shape (5,10,25) iid sampled from N(mean,cov) distribution.
        '''
        if use_psd_sqrt:
            if self.psd_generator is None:
                mean, psd_sqrt = self._compute_dataset_statistics(use_psd_sqrt=True)
                self.psd_generator = PSD_Generator(mean, psd_sqrt)
        elif self.generator is None:
            mean, cov = self._compute_dataset_statistics()
            self.generator = MultivariateNormal(mean, cov)
        else: # do nothing
            print('Generator already initialized!')
        # usage: self.generator.sample(sample_shape=torch.Size([]))
    
    def _compute_dataset_statistics(self, clamp_min=1e-6, use_psd_sqrt=False):
        x = self.dataset.x.flatten(start_dim=1)
        mean = torch.mean(x, dim=0)
        centered_data = x - mean[None, :]
        cov = torch.matmul(centered_data.T, centered_data) / (self.dataset.x.size(0) - 1)

        # sometimes, numerical instability causes cov to have negative eigenvalues
        # which breaks multivariate sampling
        if torch.min(torch.linalg.eigvalsh(cov)) > 0:
            return mean, cov
        
        if use_psd_sqrt: # to be implemented
            eigvals, eigvecs = torch.linalg.eig(cov)
            eigvals, eigvecs = eigvals.real, eigvecs.real
            clamped_eigvals = torch.clamp(eigvals, min=0)
            sqrt_eigvals = torch.sqrt(clamped_eigvals)
            psd_sqrt_cov = (eigvecs * sqrt_eigvals) @ eigvecs.T
            return mean, psd_sqrt_cov.to(dtype=mean.dtype)
        else:
            cov = cov.to(dtype=torch.float64)

            if mean.size(0) < 2948:
                eigvals, eigvecs = torch.linalg.eigh(cov)
            else:
                eigvals, eigvecs = torch.linalg.eig(cov)
                eigvals, eigvecs = eigvals.real, eigvecs.real
            
            clamped_eigvals = torch.clamp(eigvals, min=1e-6)
            cov = (eigvecs * clamped_eigvals) @ eigvecs.T # should be symmetric

            min_eigval = torch.min(torch.linalg.eigvals(cov).real)
            assert min_eigval > 0, \
                ValueError(f'Clamp min {clamp_min} insufficient, got eigenvalue {min_eigval}')
            
            return mean, cov.to(dtype=mean.dtype)
       
    def sample_dataset(self, idx=None):
        if idx is None:
            idx = torch.randint(0, len(self.dataset.x), (1,)).item()
        
        item = self.dataset.x[idx]
        
        return item.reshape(1, -1)
    
    def _supported_datasets(self): # update as tested
        return ['mnist', 'emnist', 'fmnist']
    
    def get_dataset_name(self):
        return self.dataset.name
    
    def get_input_shape(self):
        return self.dataset.x[0].squeeze().shape
    
    def evaluate(self, model, inputs=None,
                 loss_fn=None, transform=None, proj=None, add=None,
                 add_noise_std: Union[None, float, torch.Tensor] = None,
                 num_examples=None, return_logits=True):
        '''
        Expects inputs to have a batch dimension.
        '''
        if loss_fn is None:
            loss_fn = self.acc_loss

        if num_examples is None or num_examples > self.dataset_size:
            num_examples = self.dataset_size
        
        # NOTE: labels only make sense if not passing custom inputs
        # Otherwise only use evaluate as a fwd method.
        labels = self.dataset.y[:num_examples]

        if inputs is None:
            inputs = self.dataset.x[:num_examples]
        
        if transform is not None:
            inputs = transform(inputs).flatten(start_dim=1)
        else:
            inputs = inputs.flatten(start_dim=1)

        if proj is not None:
            inputs = torch.einsum('ij,bj->bi', proj, inputs)
        
        if add is not None:
            inputs = inputs + add

        if add_noise_std is not None:
            add_noise = torch.randn_like(inputs) * add_noise_std
            print(torch.norm(add_noise))
            inputs = inputs + add_noise
        
        fwd = model(inputs)

        if return_logits:
            return fwd, loss_fn(fwd, labels).item()
        else:
            return loss_fn(fwd, labels).item()
    
    def evaluate_kl_div(self, pmodel, qmodel, **kwargs):
        p, _ = self.evaluate(pmodel, **kwargs)
        q, _ = self.evaluate(qmodel, **kwargs)
        kl_div = kl_divergence(p,q).detach()
        return (kl_div.sum() / len(kl_div))
    
    def evaluate_fvu(self, pmodel, qmodel, **kwargs):
        # Evaluate the logits for both models
        p_logits, _ = self.evaluate(pmodel, **kwargs)
        q_logits, _ = self.evaluate(qmodel, **kwargs)
        
        # Calculate the variance of the pmodel logits
        total_variance = torch.var(p_logits, unbiased=False)
        
        # Calculate the residual variance (variance of the difference between pmodel and qmodel logits)
        residual_variance = torch.var(p_logits - q_logits, unbiased=False)
        
        # Calculate the fraction of variance unexplained (FVU)
        fvu = residual_variance / total_variance
        
        return fvu.item()
    

