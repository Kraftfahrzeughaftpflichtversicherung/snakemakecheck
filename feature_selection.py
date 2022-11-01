import torch
import captum.attr
import pandas as pd
from utils.utils import multiply_by_weight, set_up_weights_cell_types

def gradLoss_feature_selection(model,
                         adata,
                         method,
                         condition_key = 'study',
                         cell_type_key = 'cell_type',
                         n_epoch=2,
                         n_genes=128,
                         weighted=False,
                         path_to_params=None):
     """
     Parameters
           ----------
           model:
               CVAE model, not trained
           adata:
               AnnData object 
           method: Str
               Feature selection method: gradloss 
           condition_key: Str
               Column name which represents batch in obs in AnnData object 
           cell_type_key: Str
               Column name which represents cell type in obs in AnnData object 
           n_epoch: Int
               amount of epochs for CVAE train
           n_genes: Int
               amount of genes to select 
           weighted: Boolean
               If True - weighted, If False - non-weighted
           path_to_params: Str
               path to saved CVAE model
     
                
           Returns
           -------
               List of importnat genes, trained CVAE
     """
    
     genesets = [grad_loss,
     grad_loss_w,
     grad_loss_190th_w,
     grad_loss_190th_nw] = model.train(
                                     n_epochs=n_epoch,
                                     alpha_epoch_anneal=200,
                                    )
     genesets = list(genesets)
     for set_id in range(len(genesets)):
        genesets[set_id].index = adata.var.index
        genesets[set_id] = genesets[set_id].sort_values()[:n_genes]
     if genesets[2].sum() == 0:
        genesets[2], genesets[3] = pd.Series(), pd.Series()
     if weighted:
        return (genesets[1].index, genesets[2].index), model
     else:
        return (genesets[0].index, genesets[3].index), model
        
        
def run_methods_from_captum(model,
                          adata,
                         method,
                         condition_key = 'study',
                         cell_type_key = 'cell_type',
                         n_epoch=2,
                         n_genes=128,
                         weighted=False,
                         path_to_params=None):
     """
           Parameters
           ----------
           model:
               CVAE model, not trained
           adata:
               AnnData object 
           method: Str
               Feature selection method: integratedgradient, saliency, deeplift,
               shapleyvaluesampling
           condition_key: Str
               Column name which represents batch in obs in AnnData object 
           cell_type_key: Str
               Column name which represents cell type in obs in AnnData object 
           n_epoch: Int
               amount of epochs for CVAE train
           n_genes: Int
               amount of genes to select 
           weighted: Boolean
               If True - weighted, If False - non-weighted
           path_to_params: Str
               path to saved CVAE model
     
                
           Returns
           -------
               List of importnat genes, trained CVAE
     """
     if path_to_params is None:
        model.train(
        n_epochs=n_epoch,
        alpha_epoch_anneal=200
                  )
     else: 
        model.get_model().load_state_dict(torch.load(path_to_params))
     if method == 'integratedgradient': 
        method = captum.attr.IntegratedGradients(model.get_model().get_latent) 
    
     elif method == 'saliency': 
        method = captum.attr.Saliency(model.get_model()) 
        
     elif method == 'deeplift':  
        method = captum.attr.DeepLift(model.get_model())
     elif method == 'shapleyvaluesampling':
        method = captum.attr.ShapleyValueSampling(model.get_model())

     else:
        print('incorrect method')
        return 0
     slice_size = 800
     batch_code = model.get_batch_code(list(adata.obs.loc[:,condition_key].iloc[:slice_size])) #transform batch info into acceptable for the model form
     f_imp = pd.Series(0, index=adata.var.index)
     for i in range(list(model.get_model_sizes())[1]):
        attributions = method.attribute(torch.Tensor(adata[:slice_size].X), target=i, #target is each neuron of latent space
                                   additional_forward_args=batch_code) #batch_code is necessary to run the model 
        attrib = abs(pd.DataFrame(attributions.detach().numpy()).T)#convert to Series
        attrib.index = adata.var.index #each score now has a name of corresponding gene
        if weighted: 
            w=adata.obs.loc[:,cell_type_key].iloc[:slice_size ] #cell type info   
            attrib = multiply_by_weight(set_up_weights_cell_types(w), attrib, adata.obs.loc[:,cell_type_key].iloc[:slice_size ]) #computing weighted score
        f_imp += attrib.sum(1) #score from this particular neuron in latent space is summarized 
    
     return list(f_imp.sort_values()[-n_genes:].index), model


def run_feature_selection(model,
                          adata,
                         method,
                         condition_key = 'study',
                         cell_type_key = 'cell_type',
                         n_epoch=2,
                         n_genes=128,
                         weighted=False,
                         path_to_params=None):

     """
           Parameters
           ----------
           model:
               CVAE model, not trained
           adata:
               AnnData object 
           method: Str
               Feature selection method: integratedgradient, saliency, deeplift,
               shapleyvaluesampling, gradloss
           condition_key: Str
               Column name which represents batch in obs in AnnData object 
           cell_type_key: Str
               Column name which represents cell type in obs in AnnData object 
           n_epoch: Int
               amount of epochs for CVAE train
           n_genes: Int
               amount of genes to select 
           weighted: Boolean
               If True - weighted, If False - non-weighted
           path_to_params: Str
               path to saved CVAE model
     
                
           Returns
           -------
               List of importnat genes, trained CVAE
     """
     method = method.lower()
     methods_dict = {'gradloss': gradLoss_feature_selection,
                    'integratedgradient':run_methods_from_captum,
                    'saliency': run_methods_from_captum,
                    'deeplift': run_methods_from_captum,
                    'shapleyvaluesampling': run_methods_from_captum
     }
    
     if method.lower() not in methods_dict.keys():
        print("the method doesn't exist. Chose one of them: ", list(methods_dict.keys()))
        return 0, model
     else:
        method_run = methods_dict[method.lower()]
        return method_run(model,
                          adata,
                          method,
                          condition_key = condition_key,
                          cell_type_key =  cell_type_key,
                          n_genes=n_genes,
                          n_epoch = n_epoch, 
                          path_to_params=path_to_params,
                          weighted=weighted)
