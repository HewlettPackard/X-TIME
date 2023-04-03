###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

### Extract threshold for RF, XGB, CatBoost
import numpy as np
import re
import json
import time


def extract_rf(model,task):
    t0 = time.time()
    th_map = np.zeros((1,2*model.n_features_in_))
    th_map[:] = np.nan
    if task == 'MULTI_CLASS_CLASSIFICATION':
        leaf_value = np.zeros((1,model.n_classes_))
    elif task == 'REGRESSION':
        leaf_value = np.zeros((1,1))
    else:
        #leaf_value = np.zeros((1,1))
        leaf_value = np.zeros((1,2))
        
    tree_value = np.zeros((1,1))

    for j in range(model.n_estimators):
        if np.mod(j,np.floor(model.n_estimators/10))==0:
            print('Processing tree '+str(j/model.n_estimators)+' elapsed time '+str(time.time()-t0))
        
        estimator = model.estimators_[j]
        tree0 = estimator.tree_

        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        value = estimator.tree_.value
        #for c in range(estimator.n_classes_):
        leafNode = np.argwhere(tree0.feature == -2)
        for i in range(len(leafNode)):
            currentNode = leafNode[i]
            #leaf_value = np.vstack((leaf_value,np.argmax(value[currentNode,0,:])))
            if task == 'MULTI_CLASS_CLASSIFICATION':
                leaf_value = np.vstack((leaf_value,value[currentNode,0,:]/np.sum(value[currentNode,0,:])))
            # elif task == 'REGRESSION':    
            #     leaf_value = np.vstack((leaf_value,value[currentNode,0,:]))
            else:
                leaf_value = np.vstack((leaf_value,value[currentNode,0,:]))
            tree_value = np.vstack((tree_value,j))
            th_map_temp = np.zeros((1,2*model.n_features_in_))
            th_map_temp[:] = np.nan
            while currentNode != 0:
                if (np.argwhere(children_left == currentNode).size != 0):
                    prevNode = np.argwhere(children_left == currentNode)[0][0]
                    feature_temp = feature[prevNode]
                    threshold_temp = threshold[prevNode]
                    currentNode = prevNode
                    if np.isnan(th_map_temp[0,2*feature_temp+1]):
                        th_map_temp[0,2*feature_temp+1] = threshold_temp
                    else:
                        th_map_temp[0,2*feature_temp+1] = min(th_map_temp[0,2*feature_temp+1], threshold_temp)
                else:
                    prevNode = np.argwhere(children_right == currentNode)[0][0]
                    feature_temp = feature[prevNode]
                    threshold_temp = threshold[prevNode]
                    currentNode = prevNode
                    if np.isnan(th_map_temp[0,2*feature_temp]):
                        th_map_temp[0,2*feature_temp] = threshold_temp
                    else:
                        th_map_temp[0,2*feature_temp] = max(th_map_temp[0,2*feature_temp], threshold_temp)
            th_map = np.vstack((th_map,th_map_temp))
    th_map = np.delete(th_map,0,0)
    leaf_value = np.delete(leaf_value,0,0)
    tree_value = np.delete(tree_value,0,0)  

    #th_map = th_map.astype(np.ubyte)
    #leaf_value = leaf_value.astype(np.ubyte)
    #tree_value = tree_value.astype(np.ubyte)

    # reorganize array to make it suitable for SST
    # form [TH,LOGIT,CLASS_ID,TREE_ID]
    acam_map = np.hstack((th_map,np.hstack((leaf_value,tree_value))))
    acam_final = np.zeros([1,model.n_features_in_*2+3])
    if task =='MULTI_CLASS_CLASSIFICATION':
        n_classes = model.n_classes_
    else:
        n_classes = 2        
    for n in range(n_classes):
        acam_tmp = np.zeros([acam_map.shape[0],model.n_features_in_*2+3])
        # thresholds
        acam_tmp[:,0:model.n_features_in_*2] = acam_map[:,0:model.n_features_in_*2]
        #logits
        acam_tmp[:,model.n_features_in_*2] = acam_map[:,model.n_features_in_*2+n]
        # class ID
        acam_tmp[:,model.n_features_in_*2+1] = n*np.ones(acam_map.shape[0])
        # tree ID
        acam_tmp[:,model.n_features_in_*2+2] = acam_map[:,-1]
        acam_final = np.vstack((acam_final,acam_tmp))
    acam_final = np.delete(acam_final,0,0)
    return acam_final

def extract_xgb(model,task,data_info):
    t0 = time.time()
    # model is given as Booster
    # data_info contains the features name
    # extract dataframe
    model_df = model.trees_to_dataframe()
    # get features and convert to numbers
    features = data_info
    features_dictionary = dict(zip(features, np.linspace(0,len(features)-1,len(features)).astype(int).tolist()))
    model_df['Feature'].replace(features_dictionary, inplace=True)

    # iterate on trees
    trees_ID = np.unique(model_df['Tree'])
    acam = []
    for id in trees_ID:
        tree_df = model_df[model_df['Tree']==id]
        tree_df = tree_df.reset_index()
        # get leaves indexes
        leaves = tree_df['Feature']=='Leaf'
        leaves_idx = np.where(leaves)[0]


        # get number of features
        num_features = model.num_features()
        # get number of classes
        n_classes = int(len(np.unique(model_df['Tree']))/model.num_boosted_rounds())
        classes = np.linspace(0,n_classes-1,n_classes)
        acam_array = np.zeros((np.sum(leaves), 2*num_features+3))
        acam_array[:] = np.nan
        if np.mod(id,100)==0:
            print('Processing tree '+str(id/len(trees_ID))+' elapsed time '+str(time.time()-t0))
        l = 0
        while(l<len(leaves_idx)):
            # if np.mod(l,100)==0:
                # print('Progress '+str(l/len(leaves_idx))+'%')
            #print('Mapping leaf ', l)
            leaf_now = leaves_idx[l]
            # logit
            acam_array[l,-3] = tree_df['Gain'][leaf_now]
            # class
            acam_array[l,-2] = classes[tree_df['Tree'][leaf_now] % n_classes]
            #tree
            acam_array[l,-1] = tree_df['Tree'][leaf_now]
            # Start from the leaf and go back to the root
            current_node = tree_df['ID'][leaves_idx[l]]
            finished_branch = 0
            cat_feature_first_branch = []
            cat_feature_second_branch = []
            cat_feature_index = []
            while(finished_branch==0):    
                # parent node comes from a < type of node
                prev_node = np.where(tree_df['Yes']==current_node)[0]
                prev_node_type = 1
                if len(prev_node)==0:
                    # current node comes from a >= type of node
                    prev_node = np.where(tree_df['No']==current_node)[0]
                    prev_node_type = 0
                if len(prev_node)==0:
                    # branch finished, look at new leaf
                    l = l+1
                    finished_branch = 1
                    # copy dummy branch of the categorical feature, if any
                    if len(cat_feature_index)>0:
                        dummy_branches = np.unique(cat_feature_second_branch)
                        for dum in range(dummy_branches):
                            feature_idx = cat_feature_index[cat_feature_second_branch==dum]
                            # freeze cat features
                            freezed_feat = acam_array[dum,feature_idx]
                            # copy everything else
                            # origin branch index
                            first_branch = cat_feature_first_branch[cat_feature_second_branch==dum][0]
                            acam_array[dum,:] = acam_array[first_branch,:]
                            acam_array[dum,feature_idx] = freezed_feat
                else:
                    prev_node = tree_df[prev_node[0]:prev_node[0]+1]
                    threshold = prev_node['Split']
                    # check if it is a categorical feature
                    if str(threshold)[4] == 'N': #it is a NaN, look into cathgory
                        threshold = int(prev_node['Category'].to_list()[0][0])
                        if prev_node_type:
                            # right child                               
                            # in this case both the lower and upper bound are programmed with the threshold (special case for cathegorical feature)
                            acam_array[l,int(2*prev_node['Feature'])] = threshold
                            acam_array[l,int(2*prev_node['Feature'])+1] = threshold+1
                        else:
                            # left_child, two branches are created
                            # first one
                            new_branch = acam_array[l,:]
                            acam_array[l,int(2*prev_node['Feature'])] = threshold+1
                            # second one                         
                            acam_array = np.vstack([acam_array, new_branch])
                            acam_array[-1,int(2*prev_node['Feature'])+1] = threshold
                            # flag to copy at the end all the other features
                            cat_feature_first_branch.append(l)
                            cat_feature_second_branch.append(acam_array.shape[0])
                            cat_feature_index.append(int(2*prev_node['Feature']))

                    else:  
                        # check if a threshold is already written, in that case take the more stringent 
                        threshold_free = np.isnan(acam_array[l,int(2*prev_node['Feature']+prev_node_type)])
                        if threshold_free == 0:
                            # check if higher or smaller
                            if (threshold>acam_array[l,int(2*prev_node['Feature']+prev_node_type)]).to_list()[0]*(prev_node_type==1):
                                threshold = acam_array[l,int(2*prev_node['Feature']+prev_node_type)]
                            elif (threshold<acam_array[l,int(2*prev_node['Feature']+prev_node_type)]).to_list()[0]*(prev_node_type==0):
                                threshold = acam_array[l,int(2*prev_node['Feature']+prev_node_type)]
                        acam_array[l,int(2*prev_node['Feature']+prev_node_type)] = threshold
                    current_node = prev_node['ID'].to_list()[0]
        acam.append(acam_array)

    return np.vstack(acam)

def extract_catboost(model,task):
    n_trees = model.tree_count_
    trees_depth = model.get_tree_leaf_counts()
    n_features = len(model.feature_names_)
    n_classes = len(model.classes_)

    # save and load model
    model.save_model('tmp_cb_model.json', format='json', export_parameters=None)
    model = json.load(open('tmp_cb_model.json', "r"))

    th_low = np.zeros((1,n_features))
    th_high = np.zeros((1,n_features))
    for tree in range(n_trees):
        th_low = np.vstack((th_low, np.zeros((trees_depth[tree],n_features))))
        th_high = np.vstack((th_high, np.zeros((trees_depth[tree],n_features))))     

    th_low = np.delete(th_low,0,0)
    th_high = np.delete(th_high,0,0)
    th_low[:,:] = np.nan
    th_high[:,:] = np.nan
    n_leaves = th_low.shape[0]
    if task == 'MULTI_CLASS_CLASSIFICATION':
        leaf_value = np.zeros([n_leaves,n_classes])
    else:
        leaf_value = np.zeros([n_leaves,1])
    tree_value = np.zeros([n_leaves,1])
    for tree in range(n_trees):
        if tree == 0:
            idx1 = 0
            idx2 = trees_depth[tree] 
        else:
            idx1 = trees_depth[tree-1]
            idx2 = trees_depth[tree-1]+trees_depth[tree]
        if task == 'MULTI_CLASS_CLASSIFICATION':
            leaf_value[idx1:idx2,:] = np.array(model['oblivious_trees'][tree]['leaf_values']).reshape(trees_depth[tree],n_classes)
        else:
            leaf_value[idx1:idx2,0] = np.array(model['oblivious_trees'][tree]['leaf_values'])#.reshape(trees_depth[tree],)
        tree_value[idx1:idx2,0] = tree * np.ones(trees_depth[tree])
        for i in range(trees_depth[tree]):
            for j in range(int(np.log2(trees_depth[tree]))):
                idx = model['oblivious_trees'][tree]['splits'][j]['float_feature_index']
                th = model['oblivious_trees'][tree]['splits'][j]['border']
                if (i>>j)&1 == 1:
                    if np.isnan(th_low[idx1+i,idx]):
                        th_low[idx1+i,idx] = th
                    else:
                        th_low[idx1+i,idx] = max(th_low[idx1+i,idx],th)
                else:
                    if np.isnan(th_high[idx1+i,idx]):
                        th_high[idx1+i,idx] = th
                    else:
                        th_high[idx1+i,idx] = min(th_high[idx1+i,idx],th)

    th_map = np.empty((th_low.shape[0],th_low.shape[1]+th_high.shape[1]))
    th_map[:,::2] = th_low
    th_map[:,1::2] = th_high
    # reorganize array to make it suitable for SST
    # form [TH,LOGIT,CLASS_ID,TREE_ID]
    # Note that multiclass is not supported
    # add zeros as placeholder for classes

    acam_map = np.hstack((th_map,np.hstack((leaf_value,np.hstack((np.zeros([th_map.shape[0],1]),tree_value))))))
    # acam_final = np.zeros([1,n_features*2+3])
    # if task =='multiclass':
    #     n_classes = model.n_classes_
    # else:
    #     n_classes = 1        
    # for n in range(n_classes):
    #     acam_tmp = np.zeros([acam_map.shape[0],n_features*2+3])
    #     # thresholds
    #     acam_tmp[:,0:n_features*2] = acam_map[:,0:n_features*2]
    #     #logits
    #     acam_tmp[:,n_features*2] = acam_map[:,n_features*2+n]
    #     # class ID
    #     acam_tmp[:,n_features*2+1] = n*np.ones(acam_map.shape[0])
    #     # tree ID
    #     acam_tmp[:,n_features*2+2] = acam_map[:,-1]
    #     acam_final = np.vstack((acam_final,acam_tmp))
    # acam_final = np.delete(acam_final,0,0)
    return acam_map

def extract_catboost_old(model,task):
    t0 = time.time()
    n_trees = model.tree_count_
    trees_leaves = model.get_tree_leaf_counts()
    n_features = len(model.feature_names_)

    # save and load model
    model.save_model('tmp_cb_model.json', format='json', export_parameters=None)
    model = json.load(open('tmp_cb_model.json', "r"))
    acam_array = np.zeros([np.sum(trees_leaves),2*n_features+3])
    start_index = 0
    # iterate through the tree
    for tree in range(n_trees):
        if np.mod(tree,np.floor(n_trees/10))==0:
            print('Processing tree '+str(tree/n_trees)+' elapsed time '+str(time.time()-t0))
        # create th maps
        n_leaves = trees_leaves[tree]
        th_low = np.zeros((n_leaves,n_features))
        th_high = np.zeros((n_leaves,n_features))
        th_low[:,:] = np.nan
        th_high[:,:] = np.nan
        leaf_value = np.array(model['oblivious_trees'][tree]['leaf_values']).reshape(n_leaves,1)
        if task == 'MULTI_CLASS_CLASSIFICATION':
            class_ID = tree*np.ones((n_leaves,1))
        else:
            class_ID = np.zeros((n_leaves,1))
        tree_ID = tree*np.ones((n_leaves,1))
        for i in range(n_leaves): # leaves
            for j in range(len(model['oblivious_trees'][tree]['splits'])): #splits
                idx = model['oblivious_trees'][tree]['splits'][j]['float_feature_index']
                th = model['oblivious_trees'][tree]['splits'][j]['border']
                if (i>>j)&1 == 1:
                    if np.isnan(th_low[i,idx]):
                        th_low[i,idx] = th
                    else:
                        th_low[i,idx] = max(th_low[i,idx],th)
                else:
                    if np.isnan(th_high[i,idx]):
                        th_high[i,idx] = th
                    else:
                        th_high[i,idx] = min(th_high[i,idx],th)

        th_map = np.empty((th_low.shape[0],th_low.shape[1]+th_high.shape[1]))
        th_map[:,::2] = th_low
        th_map[:,1::2] = th_high
        leaves_class_tree = np.hstack((leaf_value,np.hstack((class_ID,tree_ID))))
        acam_tmp = np.hstack((th_map,leaves_class_tree))
        #print(acam_tmp.shape)
        end_index = start_index + trees_leaves[tree]
        acam_array[start_index:end_index,:] = acam_tmp
        start_index = end_index*1

    return acam_array

def extract_thresholds(model,algorithm,task,data_info = 0):
    
    if algorithm =='rf':
        cam_map = extract_rf(model,task)
    elif algorithm == 'xgboost':
        # if feature name not relevant, autogenerated
        # otherwise it should be list of the feature names
        if np.sum(data_info == 0):
            data_info_helper = np.linspace(0,model.num_features()-1,model.num_features())
            data_info = []
            for d in data_info_helper:
                data_info.append('f'+str(int(d)))
        cam_map = extract_xgb(model,task,data_info)    
    elif algorithm == 'catboost':
        cam_map = extract_catboost_old(model, task)
    elif algorithm == 'lightgbm':
        print('Currently not supported')
        cam_map = None
        # features_count = model.n_features_
        # onnx_model = onnxmltools.convert_lightgbm(model, name='LightGBM', initial_types=[['input', FloatTensorType([0, features_count])]])
        # onnxmltools.utils.save_model(onnx_model, 'model.onnx')
        # catboost_model = CatBoostClassifier()
        # catboost_model.load_model('model.onnx', format='onnx')
        # cam_map = extract_catboost(catboost_model, task)

    return cam_map

def map_to_ubyte(cam_map,X_test,n_bits = 8):
    th = cam_map[:,0:2*X_test.shape[1]]
    # Find vins for each feature
    # bins = []
    # for i in range(0,th.shape[1],1):
    #     all_th = th[:,i:i+1].reshape(-1)
    #     bins.append(np.unique(all_th[np.where(np.isnan(all_th)==0)[0]]))
    # for i in range(X_test.shape[1]):
    #     X_test[:,i] = np.digitize(X_test[:,i], bins[i],right=True).astype(np.ubyte)

    for i in range(X_test.shape[1]):
        th_sel = th[:,2*i:2*i+2]
        bins = np.unique(th_sel)[~np.isnan(np.unique(th_sel))]
        if len(bins>2**n_bits):
            bins= np.linspace(np.nanmin(th_sel.reshape(-1))*(np.sign(np.nanmin(th_sel.reshape(-1)))*(-0.01)+1.), np.nanmax(th_sel.reshape(-1))*1.01, 2**n_bits+1)
        nan_indexes = np.where(np.isnan(cam_map[:,2*i]))[0]
        cam_map[:,2*i] = np.digitize(th_sel[:,0],bins).astype(np.ubyte)
        cam_map[nan_indexes,2*i]=np.nan
        nan_indexes = np.where(np.isnan(cam_map[:,2*i+1]))[0]
        cam_map[:,2*i+1] = np.digitize(th_sel[:,1],bins).astype(np.ubyte)
        cam_map[nan_indexes,2*i+1]=np.nan
        X_test[:,i] = np.digitize(X_test[:,i], bins,right=False).astype(np.ubyte)


    return cam_map,X_test+0.5

