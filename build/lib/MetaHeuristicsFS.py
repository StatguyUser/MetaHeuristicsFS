"""
@author: Md Azimul Haque
"""

import numpy as np
np.random.seed(1)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold

from math import ceil
from copy import deepcopy
import time
import warnings
import sys
import os
if not sys.warnoptions:    
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = 'ignore'
    
    
def create_formatted_data(dataset,dependent_variable_name,features_list,stratified=False,split_ratio = 0.2,k_folds=5):
    
    '''
    Input Parameters
    ----------------

    dataset : Pandas dataframe which has all the feaures and dependent variable
    
    dependent_variable_name : The name of dependent variable, as a string. For example, 'salary'
    
    features_list : Python list with all the feature names
    
    stratified : If you will like the data split to be stratified. Default value is False.
    
    split_ratio : ratio in which you will like to split train and test data. Default is 0.2
    
    k_folds : number of k-fold cross-validations. Default is 5.
    

    Output Values
    -------------

    data_dict : Dictionary object with number of keys equal to number of k_folds. Each has train and test data
    
    x_validation : Feature matrix for validation data
    
    y_validation : Dependent variable for validation data, as array.
    
    x_external_test : Feature matrix for external test data
    
    y_external_test : Dependent variable for external test data, as array.
    
    
    '''
    
    #train, and external test
    if stratified:
        data_intermediate,x_external_test,y_intermediate,y_external_test = train_test_split(dataset,dataset[[dependent_variable_name]],stratify=dataset[[dependent_variable_name]],test_size=split_ratio)
    else:
        data_intermediate,x_external_test,y_intermediate,y_external_test = train_test_split(dataset,dataset[[dependent_variable_name]],test_size=split_ratio)
    
    data_intermediate.reset_index(inplace=True,drop=True)
    x_external_test.reset_index(inplace=True,drop=True)
    y_intermediate.reset_index(inplace=True,drop=True)
    y_external_test.reset_index(inplace=True,drop=True)

    dataset[[dependent_variable_name]]
    
    #train and validation data
    data_train,x_validation,y_train,y_validation = train_test_split(data_intermediate,y_intermediate,stratify=y_intermediate,test_size=split_ratio)
    data_train.reset_index(inplace=True,drop=True)
    x_validation.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_validation.reset_index(inplace=True,drop=True)
    
    data_dict = {}
    
    if stratified:

        #stratified k fold cross validation    
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)        
        index = 0
        
        for train_index, test_index in skf.split(data_train, y_train):
            data_dict[index]={'x_train':data_train[data_train.index.isin(train_index)][features_list],
                              'y_train':data_train[data_train.index.isin(train_index)][[dependent_variable_name]],
                              'x_test':data_train[data_train.index.isin(test_index)][features_list],
                              'y_test':data_train[data_train.index.isin(test_index)][[dependent_variable_name]]}
            index += 1
    else:
        
        #stratified k fold cross validation    
        skf = KFold(n_splits=k_folds, shuffle=True, random_state=1)        
        index = 0

        for train_index, test_index in skf.split(data_train):
            data_dict[index]={'x_train':data_train[data_train.index.isin(train_index)][features_list],
                              'y_train':data_train[data_train.index.isin(train_index)][[dependent_variable_name]],
                              'x_test':data_train[data_train.index.isin(test_index)][features_list],
                              'y_test':data_train[data_train.index.isin(test_index)][[dependent_variable_name]]}
            index += 1
            
    return data_dict,x_validation,y_validation,x_external_test,y_external_test

    
class ParticleSwarmOptimizationFS:
    
    '''
    Machine Learning Parameters
    ----------

    columns_list : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.

    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}
        
        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.
    
    x_validation_dataframe : dataframe containing features of validatoin dataset
    
    y_validation_dataframe : dataframe containing dependent variable of validation dataset
    
    model : Model object. It should have .fit and .predict attribute
        
    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'

    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
    
    Particle Swarm Optimization Parameters
    ----------
    
    iterations : Number of times particle swarm optimization will search for solutions. Default is 100
    
    swarmSize : Size of the swarm in each iteration. Default is 100.

    run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.

    Output
    ----------
    best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
        
    '''

    def __init__(self,columns_list,data_dict,x_validation_dataframe,y_validation_dataframe,model,cost_function,cost_function_improvement='increase',average=None,iterations=100,swarmSize=100,run_time=120,print_iteration_result = True):
        self.columns_list=columns_list
        self.data_dict=data_dict
        self.model=model
        self.x_validation_dataframe=x_validation_dataframe
        self.y_validation_dataframe=y_validation_dataframe
        self.cost_function=cost_function
        self.cost_function_improvement=cost_function_improvement
        self.average=average

        self.iterations = iterations
        self.swarmSize = swarmSize
        self.run_time = run_time
        self.print_iteration_result = print_iteration_result

        #extra
        self.MaxVelocity = 1
        self.MinVelocity = -1
        self.c1 = 0
        self.c2 = 0
        self.particles = []
        self.velocity = []
        self.pBest = []
        self.gBest = 0
        self.gBestFeatures = []
        self.pBestFeatures = []
        
    def _initialize_position(self):

        ## create binary matrix with swarm size and number of features as dimension

        ##output is swarm, with number of rows equal to swarm size, in other words index i = 1 to s
        
        self.particles = np.random.randint(0, 2, size=(self.swarmSize, len(self.columns_list)))
        
        self.pBestFeatures = np.zeros((self.swarmSize, len(self.columns_list)))
    
    def _initialize_velocity(self):

        ## create matrix with swarm size and number of features as dimension
                
        upper = int(round(len(self.columns_list)/3))+1
        
        self.velocity = np.random.randint(1, upper, size=(self.swarmSize, len(self.columns_list)))

    
    def _get_feature_names(self,binary_index):
        
        feature_names = []
        
        for i in range(len(self.columns_list)):
            if binary_index[i] == 1:
                feature_names.append(self.columns_list[i])
        
        return feature_names

        

    def _calculate_cost(self,current_at_feature_subset):
    
        fold_cost = []
        
        for i in self.data_dict.keys():
    
            x_train=self.data_dict[i]['x_train'][current_at_feature_subset]
            y_train=self.data_dict[i]['y_train']
            
            x_test=self.data_dict[i]['x_test'][current_at_feature_subset]
            y_test=self.data_dict[i]['y_test']
            
            self.model.fit(x_train,y_train)

            y_test_predict=self.model.predict(x_test)

            y_validation_predict=self.model.predict(self.x_validation_dataframe[current_at_feature_subset])
            
            if self.average:
                fold_cost.append(self.cost_function(y_test,y_test_predict,average=self.average))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict,average=self.average))
            else:
                fold_cost.append(self.cost_function(y_test,y_test_predict))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict))

        return np.mean(fold_cost)


    def _calc_intertia(self,iter_num,w_old):
        return ((w_old - 0.4) * (self.iterations - iter_num))/(self.iterations + 0.4)
    
    def _get_constants(self):

        ## Effects of Random Values for Particle Swarm Optimization Algorithm
        
        sum_val = 3
        array = []
        
        while (sum_val > 4.5 or sum_val < 4) and (self.c1 < 1.9 or self.c1 > 2.2) and (self.c2 < 1.9 or self.c2 > 2.2):
            array = np.random.uniform(low=1.3, high=2.7, size=2)
            self.c1, self.c2 = array[0], array[1]
            sum_val = self.c1+self.c2
            
    
    def _update_position_velocity(self,w_inertia,particle_index,element_index):

        new_velocity = 0
        
        currentParticle = self.particles[particle_index][element_index]
        localBestParticle = self.pBestFeatures[particle_index][element_index]
        globalBestParticle = self.gBestFeatures[element_index]
        
        currentVelocity = self.velocity[particle_index][element_index]

        r1 = np.random.rand()
        r2 = np.random.rand()
        
        #calculate new velocity
        new_velocity = (w_inertia * currentVelocity) + (self.c1 * r1 * (localBestParticle - currentParticle)) + (self.c2 * r2 * (globalBestParticle - currentParticle))
        
        #check against MaxVelocity and MinVelocity
        if (new_velocity > self.MaxVelocity):
            new_velocity = deepcopy(self.MaxVelocity)
        if(new_velocity < self.MinVelocity):
            new_velocity = deepcopy(self.MinVelocity)
        
        #assign new velocity
        self.velocity[particle_index][element_index] = new_velocity
        
        #calculate new position
        currentParticle += new_velocity
        
        #normalize value of currentParticle as eother 0 or 1
        if currentParticle > 1:
            self.particles[particle_index][element_index] = 1
        elif currentParticle < 0:
            self.particles[particle_index][element_index] = 0
        else:
            self.particles[particle_index][element_index] = int(round(currentParticle))


    def GetBestFeatures(self):

        if self.cost_function_improvement == 'decrease':
            #gBest is global best, pBest is particle best for each individual particle
            self.pBest = np.array([np.inf]*self.swarmSize)
            self.gBest = np.inf
        else:
            self.pBest = np.zeros(self.swarmSize)
            self.gBest = 0
        
        #get starting time
        start = time.time()
        
        ### MaxVelocity decided maximum limit of number of features that will be changed.
        if self.MaxVelocity == 0:
            self.MaxVelocity = int(round(len(self.columns_list)/3))
        elif self.MaxVelocity > len(self.columns_list):
            self.MaxVelocity = len(self.columns_list)
        
        #first iteration, default value
        w_inertia = 0.5
        
        # c1 and c2 get values
        self._get_constants()
        
        ### create randomly initialized particle positions

        self._initialize_position()
        
        ### initialize velocity, same dimension as particle

        self._initialize_velocity()
        
        ### For each iteratoin of PSO
        for iter_num in range(self.iterations):
            
            #check if time exceeded
            if (time.time()-start)//60>self.run_time:
                print('================= Run time exceeded allocated time. Producing best solution generated so far. =================')
                break

            #from second iteration, take the dynamic w
            if iter_num > 0:
                w_inertia = self._calc_intertia(iter_num,w_inertia)
            
            
            #for each particle
            for particle in range(self.swarmSize):
                #get particle
                feature_names = self._get_feature_names(self.particles[particle])

                #get cost for particle. Swap with pBest and gBest, if new cost is better
                cost = self._calculate_cost(feature_names)
                
                if self.cost_function_improvement == 'decrease':
                    if cost < self.gBest:
                        self.gBest = deepcopy(cost)
                        self.gBestFeatures = deepcopy(self.particles[particle])
                        
                    if cost < self.pBest[particle]:
                        self.pBest[particle] = cost
                        self.pBestFeatures[particle] = deepcopy(self.particles[particle])
                else:
                    if cost > self.gBest:
                        self.gBest = deepcopy(cost)
                        self.gBestFeatures = deepcopy(self.particles[particle])
                        
                    if cost > self.pBest[particle]:
                        self.pBest[particle] = cost
                        self.pBestFeatures[particle] = deepcopy(self.particles[particle])

            #for each particle
            for particle_index in range(self.swarmSize):
                for element_index in range(len(self.columns_list)):
                    ### Update position and velocity
                    self._update_position_velocity(w_inertia,particle_index,element_index)
            if self.print_iteration_result:
                print('Best combined performance on test and validation data for iteration '+str(iter_num)+': '+str(self.gBest))

        ## After everything is done, just get the original name of feature.
        best_columns = []
        for indx in range(len(self.gBestFeatures)):
            if self.gBestFeatures[indx] == 1:
                best_columns.append(self.columns_list[indx])
            

        print('================= Best result:',self.gBest,'=================')
        print('================= Execution time in minutes:',(time.time()-start)//60,'=================')

        return best_columns

class AntColonyOptimizationFS:
    
    '''
    Machine Learning Parameters
    ----------

    columns_list : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.

    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}
        
        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.
    
    x_validation_dataframe : dataframe containing features of validatoin dataset
    
    y_validation_dataframe : dataframe containing dependent variable of validation dataset
        
    model : Model object. It should have .fit and .predict attribute
        
    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'

    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
    
    Ant Colony Optimization Parameters
    ----------
    
    iterations : Number of times ant colony optimization will search for solutions. Default is 100
    
    N_ants : Number of ants in each iteration. Default is 100.

    run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.

    evaporation_rate : Evaporation rate. Values are between 0 and 1. If it is too large, chances are higher to find global optima, but computationally expensive. If it is low, chances of finding global optima are less. Default is kept as 0.9
    
    Q : Pheromene update coefficient. Value between 0 and 1. It affects the convergence speed. If it is large, ACO will get stuck at local optima. Default is kept as 0.2

    Output
    ----------
    best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
        
    '''

    def __init__(self,columns_list,data_dict,x_validation_dataframe,y_validation_dataframe,model,cost_function,cost_function_improvement='increase',average=None,iterations=100,N_ants=100,run_time=120,print_iteration_result = True,evaporation_rate=0.9,Q=0.2):
        self.columns_list=columns_list
        self.data_dict=data_dict
        self.x_validation_dataframe=x_validation_dataframe
        self.y_validation_dataframe=y_validation_dataframe
        self.model=model
        self.cost_function=cost_function
        self.cost_function_improvement=cost_function_improvement
        self.average=average
        self.iterations = iterations
        self.N_ants = N_ants
        self.run_time = run_time
        self.print_iteration_result = print_iteration_result
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        self.fp = [1]*(len(columns_list))
        self.ants = []
        self.size = len(columns_list)
        self.topScore = 0
        self.result=[]

    def _calculate_cost(self,current_at_feature_subset):
    
        fold_cost = []
        
        for i in self.data_dict.keys():
    
            x_train=self.data_dict[i]['x_train'][current_at_feature_subset]
            y_train=self.data_dict[i]['y_train']
            
            x_test=self.data_dict[i]['x_test'][current_at_feature_subset]
            y_test=self.data_dict[i]['y_test']
            
            self.model.fit(x_train,y_train)

            y_test_predict=self.model.predict(x_test)

            y_validation_predict=self.model.predict(self.x_validation_dataframe[current_at_feature_subset])
            
            if self.average:
                fold_cost.append(self.cost_function(y_test,y_test_predict,average=self.average))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict,average=self.average))
            else:
                fold_cost.append(self.cost_function(y_test,y_test_predict))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict))

        return np.mean(fold_cost)


    def _constructAntSolution(self,ant):
        
        current_at_feature_subset = []   
        
        featureSetIndex = []
        #for each feature index
        for j in range(self.size):

            #generate random number
            decision = np.random.rand()

            ## in the first iteration, fp has all values as 1 for the length of features

            if decision < self.fp[j] / 2.0:
                featureSetIndex.append(1)
                current_at_feature_subset.append(self.columns_list[j])
            else:
                featureSetIndex.append(0)
        
        # if no features, then just keep 0.5. Else, calculate the actual cost
        if sum(featureSetIndex) == 0:
            score = 0.5
        else:
            score = float(self._calculate_cost(current_at_feature_subset))
        
        #for the ant, assign score and feature indexes used.
        ant.val = score
        ant.subsets = deepcopy(featureSetIndex)
        
        return ant

    def _ApplyLocalSearch(self):

        maxSet = []
        
        if self.cost_function_improvement == 'decrease':
            maxScore = np.inf
        else:
            maxScore = 0
        
        #for each ant in the iteration
        for a in self.ants:

            if self.cost_function_improvement == 'decrease':
                if maxScore > a.val or (maxScore == a.val and (maxSet and sum(a.subsets) < sum(maxSet))):
                    maxScore = a.val
                    maxSet = a.subsets
            else:
                if maxScore < a.val or (maxScore == a.val and (maxSet and sum(a.subsets) < sum(maxSet))):
                    maxScore = a.val
                    maxSet = a.subsets

        ## After the search for best score is done and associated feature set binary vector is found, 

        if self.cost_function_improvement == 'decrease':
            if self.topScore > maxScore or (maxScore == self.topScore and (self.result and sum(maxSet) < sum(self.result))):
                self.topScore = maxScore
                self.result = maxSet
        else:
            if self.topScore < maxScore or (maxScore == self.topScore and (self.result and sum(maxSet) < sum(self.result))):
                self.topScore = maxScore
                self.result = maxSet
        
        ##but return only local best result for current colony
        return maxSet, maxScore

    
    def _calc_update_param(self,topScore):
        
        
        sumResults = 0
        
        for a in self.ants:

            if sum(a.subsets) > 0:
                #value that is added is pehermone update coefficient divided by cost, for both current and top ant
                sumResults += (self.Q/a.val)
        
        return sumResults + (self.Q/topScore)

    
    def _UpdatePheromoneTrail(self,topSet, topScore):
        
        #get sum results
        sumResults = self._calc_update_param(topScore)
        
        #topSet is binary 1|0 feature vector. topScore is best score for entire colony
        
        for i,v in enumerate(topSet):

            #evaporate pheromene, based on formula 𝜏𝑖 = 𝜌 ∗ 𝜏𝑖

            pheromone_at_index = self.fp[i]*self.evaporation_rate
            
            ## update pheromene trail

            if v == 1:
                pheromone_at_index += self.fp[i] + sumResults
            
            self.fp[i] = pheromone_at_index

    
    def GetBestFeatures(self):
        
        if self.cost_function_improvement == 'decrease':
            self.topScore = np.inf
        
        #get starting time
        start = time.time()
        
        ### For each iteratoin of ACO
        for iter_num in range(self.iterations):
            
            #check if time exceeded
            if (time.time()-start)//60>self.run_time:
                print('================= Run time exceeded allocated time. Producing best solution generated so far. =================')
                break
    
            #for each ant
            for i in range(self.N_ants):
                #create new ant
                ant = Ant()
                #create the first initialization for ant
                ant = self._constructAntSolution(ant)
                self.ants.append(ant)
            
            ##for the iteration, after all colony of ants have been created

            topSet, topScore = self._ApplyLocalSearch()
            if self.print_iteration_result:
                print('Best combined performance on test and validation data for iteration '+str(iter_num)+': '+str(self.topScore))
            
            #give input the best feature binary 1|0 vector and best metric from the entire colony
            self._UpdatePheromoneTrail(topSet, topScore)
            self.ants = []

        ## After everything is done, just get the original name of feature.
        best_columns = []
        for indx in range(len(self.result)):
            if self.result[indx] == 1:
                best_columns.append(self.columns_list[indx])
        

        print('================= Best result:',self.topScore,'=================')
        print('================= Execution time in minutes:',(time.time()-start)//60,'=================')

        return best_columns

class Ant:
    def __init__(self):
        self.subsets = []
        self.val = 0

class SimulatedAnnealingFS:
    '''
    Machine Learning Parameters
    ----------

    columns_list : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.

    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}
        
        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.
    
    x_validation_dataframe : dataframe containing features of validatoin dataset
    
    y_validation_dataframe : dataframe containing dependent variable of validation dataset
        
    model : Model object. It should have .fit and .predict attribute
        
    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'

    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
    
    Simulated Annealing Parameters
    ----------
    
    temperature : Initial temperature for annealing. Default is 1500
    
    iterations : Number of times simulated annealing will search for solutions. Default is 100
    
    n_perturb : Number of times feature set will be perturbed in an iteration. Default is 1.

    n_features_percent_perturb : Percentage of features that will be perturbed during each perturbation. Value are between 1 and 100.
    
    alpha : Temperature reduction factor. Defaults is 0.9

    run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.

    Output
    ----------
    best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
        
    '''
    
    def __init__(self,columns_list,data_dict,x_validation_dataframe,y_validation_dataframe,model,cost_function,cost_function_improvement='increase',average=None,temperature=1500,iterations=100,n_perturb=10,run_time=120,print_iteration_result = True,n_features_percent_perturb=1,alpha=0.9):
        self.columns_list=columns_list
        self.data_dict=data_dict
        self.x_validation_dataframe=x_validation_dataframe
        self.y_validation_dataframe=y_validation_dataframe
        self.model=model
        self.cost_function=cost_function
        self.cost_function_improvement=cost_function_improvement
        self.average=average
        self.temperature=temperature
        self.iterations=iterations
        self.n_perturb=n_perturb
        self.run_time=run_time
        self.print_iteration_result = print_iteration_result
        self.n_features_percent_perturb=n_features_percent_perturb
        self.alpha=alpha
        self.topScore = 0
        self.result=[]
        
    def _calculate_cost(self,current_subset):
    
        fold_cost = []
        
        for i in self.data_dict.keys():
    
            x_train=self.data_dict[i]['x_train'][current_subset]
            y_train=self.data_dict[i]['y_train']
            
            x_test=self.data_dict[i]['x_test'][current_subset]
            y_test=self.data_dict[i]['y_test']
            
            self.model.fit(x_train,y_train)
            
            y_test_predict=self.model.predict(x_test)
            y_validation_predict=self.model.predict(self.x_validation_dataframe[current_subset])
            
            if self.average:
                fold_cost.append(self.cost_function(y_test,y_test_predict,average=self.average))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict,average=self.average))
            else:
                fold_cost.append(self.cost_function(y_test,y_test_predict))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict))
                
        return np.mean(fold_cost)

    def GetBestFeatures(self):
    
        Temp = []
        Min_Cost = []
        current_subset = []
        
        n_features_perturb = ceil((len(self.columns_list)/100)*self.n_features_percent_perturb)
        
        if self.cost_function_improvement == 'decrease':
            self.topScore = np.inf
        
        start = time.time()
        
        ### For each iteratoin of SA        
        for iter_num in range(self.iterations):
            
            if (time.time()-start)//60>self.run_time:
                print('================= Run time exceeded allocated time. Producing best solution generated so far. =================')
                break
        
            #in each iteration, start with a subset, then perturn
            
            if iter_num==0:
                ### Create an current_subset of features
                ones = np.random.randint(0,2,len(self.columns_list))
                for feature_index in range(len(self.columns_list)):
                    if ones[feature_index] == 1:
                        current_subset.append(self.columns_list[feature_index])
            
                ### Calculate cost function for current_subset of features
                cost_current_subset = self._calculate_cost(current_subset)
            else:
                cost_current_subset = self._calculate_cost(current_subset)
            
            for j in range(self.n_perturb):
                ### START PERTURB
            
                perturb_index = []
    
                while len(perturb_index) <= n_features_perturb:
                    #random number between 0 and length of current_subset
                    rand_index = np.random.randint(0,len(self.columns_list))
                    if rand_index not in perturb_index:
                        perturb_index.append(rand_index)
                        
                ##perturb by checking if indexes are present in current subset, then dont add, else add
                perturb_subset = deepcopy(current_subset)
                for p_index in perturb_index:
                    #if feature at current index present, remove; else add
                    if self.columns_list[p_index] in current_subset:
                        perturb_subset.remove(self.columns_list[p_index])
                    else:
                        perturb_subset.append(self.columns_list[p_index])
                
                #calculate cost for perturbed feature set            
                cost_perturb_subset = self._calculate_cost(perturb_subset)
        
                rand_prob = np.random.rand()
                accept_reject_prob = 1/(np.exp(cost_perturb_subset-cost_current_subset)/self.temperature)
                
                ### if cost_new is better than old solution, swap new solution with initial solution
                ### if rand1 small than calculated 'form', swap new solution with initial solution
                ### else keep the initial solution
                
                if self.cost_function_improvement == 'decrease':
                    #if cost_new is better than old solution, swap new solution with initial solution
                    if cost_perturb_subset < cost_current_subset:
                        current_subset = deepcopy(perturb_subset)
                    #if performance is bad, then check accept_reject_prob is rand1 small than calculated 'form', swap new solution with initial solution
                    elif rand_prob < accept_reject_prob:
                        current_subset = deepcopy(perturb_subset)
        
                else:
                    #if cost_new is better than old solution, swap new solution with initial solution
                    if cost_perturb_subset > cost_current_subset:
                        current_subset = deepcopy(perturb_subset)
                    #if rand1 small than calculated 'form', swap new solution with initial solution
                    elif rand_prob > accept_reject_prob:
                        current_subset = deepcopy(perturb_subset)
        
            
            ### append initial cost at end of loop, save initial cost and recalculate T0. 
            Temp.append(self.temperature)


            if self.cost_function_improvement == 'decrease':
                if self.topScore > cost_perturb_subset:
                    
                    self.topScore = deepcopy(cost_perturb_subset)
                    self.result = deepcopy(perturb_subset)
                    
                elif self.topScore > cost_current_subset:
                    
                    self.topScore = deepcopy(cost_current_subset)
                    self.result = deepcopy(current_subset)
            else:
                if self.topScore < cost_perturb_subset:
                    
                    self.topScore = deepcopy(cost_perturb_subset)
                    self.result = deepcopy(perturb_subset)
                    
                elif self.topScore < cost_current_subset:
                    
                    self.topScore = deepcopy(cost_current_subset)
                    self.result = deepcopy(current_subset)

            
            if self.cost_function_improvement == 'decrease':
                if cost_current_subset < cost_perturb_subset:
                    Min_Cost.append(cost_current_subset)

                elif cost_current_subset > cost_perturb_subset:
                    Min_Cost.append(cost_perturb_subset)
                    current_subset = deepcopy(perturb_subset)
            else:
                if cost_current_subset < cost_perturb_subset:
                    Min_Cost.append(cost_perturb_subset)
                    current_subset = deepcopy(perturb_subset)
                elif cost_current_subset > cost_perturb_subset:
                    Min_Cost.append(cost_current_subset)

            if self.print_iteration_result:
                print('Best combined performance on test and validation data for iteration '+str(iter_num)+': '+str(Min_Cost[-1]))
            
            self.temperature = self.alpha*self.temperature

        print('================= Best result:',self.topScore,'=================')
        print('================= Execution time in minutes:',(time.time()-start)//60,'=================')

        return self.result

class GeneticAlgorithmFS:
    '''
    Machine Learning Parameters
    ----------
    
    model : Model object. It should have .fit and .predict attribute
        
    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}
        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.

    x_validation_dataframe : dataframe containing features of validatoin dataset
    
    y_validation_dataframe : dataframe containing dependent variable of validation dataset
        
    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
    
    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'
    
    columns_list : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.
    
    Genetic Algorithm Parameters
    ----------
    
    generations : Number of generations to run genetic algorithm. 100 as deafult
    
    population : Number of individual chromosomes. 50 as default. It should be kept as low number if number of possible permutation and combination of feature sets are small.
    
    prob_crossover : Probability of crossover. 0.9 as default
    
    prob_mutation : Probability of mutation. 0.1 as default
        
    run_time : Number of minutes to run the algorithm. This is checked in between generations.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from generations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.
    Output
    ----------
    best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
    
    
    '''
    def __init__(self,model,data_dict,x_validation_dataframe,y_validation_dataframe,cost_function,average=None,cost_function_improvement='increase',columns_list=[],generations=50,population=40,prob_crossover=0.9,prob_mutation=0.1,run_time=120,print_iteration_result = True):
        self.model=model
        self.data_dict=data_dict
        self.x_validation_dataframe=x_validation_dataframe
        self.y_validation_dataframe=y_validation_dataframe
        self.cost_function=cost_function
        self.average=average
        self.cost_function_improvement=cost_function_improvement
        self.generations=generations
        self.population=population
        self.prob_crossover=prob_crossover
        self.prob_mutation=prob_mutation
        self.run_time=run_time
        self.print_iteration_result = print_iteration_result
        self.columns_list=columns_list


    def _get_feature_index(self,features):
        t=0
        index_list=[]
        
        for feat in features:
            if feat==1:
                index_list.append(t)
            t+=1
        return index_list
    
    def _getModel(self):
        return self.model
    
    def _getCost(self,population_array):
        
        columns_list=list(map(list(self.columns_list).__getitem__,self._get_feature_index(population_array)))

        fold_cost=[]
        
        for i in self.data_dict.keys():
            
            x_train=self.data_dict[i]['x_train'][columns_list]
            y_train=self.data_dict[i]['y_train']
            
            x_test=self.data_dict[i]['x_test'][columns_list]
            y_test=self.data_dict[i]['y_test']
            
            model=self._getModel()
            
            model.fit(x_train,y_train)
            y_test_predict=model.predict(x_test)
            y_validation_predict=self.model.predict(self.x_validation_dataframe[columns_list])
            
            if self.average:
                fold_cost.append(self.cost_function(y_test,y_test_predict,average=self.average))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict,average=self.average))
            else:
                fold_cost.append(self.cost_function(y_test,y_test_predict))
                fold_cost.append(self.cost_function(self.y_validation_dataframe,y_validation_predict))

        return np.mean(fold_cost)

    
    def _check_unmatchedrows(self,population_matrix,population_array):
        pop_check=0
        
        #in each row of population matrix
        for pop_so_far in range(population_matrix.shape[0]):
            #check duplicate
            if sum(population_matrix[pop_so_far]!=population_array)==population_array.shape[0]:
                #assign 1 for duplicate
                pop_check=1
                break
        return pop_check
    def _get_population(self,population_matrix,population_array):
        iterate=0
        ## append until population and no duplicate chromosome
        while population_matrix.shape[0]<self.population:
            #prepare population matrix
            np.random.shuffle(population_array)
            
            #check if it is first iteration, if yes then append
            if iterate==0:
                population_matrix=np.vstack((population_matrix,population_array))
                iterate+=1
            #if second iteration and once chromosome already, check if it is duplicate
            elif iterate==1 and sum(population_matrix[0]==population_array)!=population_array.shape[0]:
                population_matrix=np.vstack((population_matrix,population_array))
                iterate+=1
            elif iterate>1 and self._check_unmatchedrows(population_matrix,population_array)==0:
                population_matrix=np.vstack((population_matrix,population_array))
        
        return population_matrix
    
    def _get_parents(self,population_array,population_matrix):
        
        #keep space for best chromosome
        parents = np.empty((0,population_array.shape[0]))
        
        #get 6 unique index to fetch from population
        indexes=np.random.randint(0,population_matrix.shape[0],6)
        
        while len(np.unique(indexes))<6:
            indexes=np.random.randint(0,population_matrix.shape[0],6)
            
        #mandatory run twice as per GA algorithm
        for run_range in range(2):
            
            #get 3 unique index to fetch from population
            if run_range==0:
                index_run=indexes[0:3]
            #if second run then from half till end
            else:
                index_run=indexes[3:]
                
            ## gene pool 1
            gene_1=population_matrix[index_run[0]]
            ## cost of gene 1
            fold_cost1=self._getCost(population_array=gene_1)

            ## gene pool 2
            gene_2=population_matrix[index_run[1]]
            ## cost of gene 2
            fold_cost2=self._getCost(population_array=gene_2)
            
            ## gene pool 3
            gene_3=population_matrix[index_run[2]]
            ## cost of gene 2
            fold_cost3=self._getCost(population_array=gene_3)

            if self.cost_function_improvement=='increase':            
                #get best chromosome from 3 and assign best chromosome
                if fold_cost1==max(fold_cost1,fold_cost2,fold_cost3):
                    parents=np.vstack((parents,gene_1))
                elif fold_cost2==max(fold_cost1,fold_cost2,fold_cost3):
                    parents=np.vstack((parents,gene_2))
                elif fold_cost3==max(fold_cost1,fold_cost2,fold_cost3):
                    parents=np.vstack((parents,gene_3))
            elif self.cost_function_improvement=='decrease':
                #get best chromosome from 3 and assign best chromosome
                if fold_cost1==min(fold_cost1,fold_cost2,fold_cost3):
                    parents=np.vstack((parents,gene_1))
                elif fold_cost2==min(fold_cost1,fold_cost2,fold_cost3):
                    parents=np.vstack((parents,gene_2))
                elif fold_cost3==min(fold_cost1,fold_cost2,fold_cost3):
                    parents=np.vstack((parents,gene_3))
                                
        return parents[0],parents[1]
    
    def _crossover(self,parent1,parent2):
        
        #placeholder for child chromosome
        child1=np.empty((0,len(parent1)))
        child2=np.empty((0,len(parent2)))
        
        crsvr_rand_prob=np.random.rand()
        
        if crsvr_rand_prob < self.prob_crossover:
            while np.sum(child1)==0 or np.sum(child2)==0:
                ##initiate again
                child1=np.empty((0,len(parent1)))
                child2=np.empty((0,len(parent2)))
        
                index1=np.random.randint(0,len(parent1))
                index2=np.random.randint(0,len(parent2))
                
                #get different indices to make sure crossover happens
                while index1 == index2:
                    index2=np.random.randint(0,len(parent1))
                    
                index_parent1=min(index1,index2)
                index_parent2=max(index1,index2)
                
                #parent1
                #first segment
                first_seg_parent1=parent1[:index_parent1]
                #second segment
                mid_seg_parent1=parent1[index_parent1:index_parent2+1]
                #third segment
                last_seg_parent1=parent1[index_parent2+1:]
                child1=np.concatenate((first_seg_parent1,mid_seg_parent1,last_seg_parent1))
                
                #parent2
                #first segment
                first_seg_parent2=parent2[:index_parent1]
                #second segment
                mid_seg_parent2=parent2[index_parent1:index_parent2+1]
                #third segment
                last_seg_parent2=parent2[index_parent2+1:]
                child2=np.concatenate((first_seg_parent2,mid_seg_parent2,last_seg_parent2))
            
            return child1,child2
        else:
            return parent1,parent2
        
    def _mutation(self,child):
        #mutated child 1 placeholder
        mutated_child=np.empty((0,len(child)))
        
        while np.sum(mutated_child)==0:
            mutated_child=np.empty((0,len(child)))

            #get random probability at each index of chromosome and start with 0
            t=0
            
            for cld1 in child:
                rand_prob_mutation = np.random.rand()
                if rand_prob_mutation<self.prob_mutation:
                    #swap value
                    if child[t]==0:
                        child[t]=1
                    else:
                        child[t]=0
                    
                    mutated_child=child
                #if probability is less
                else:
                    mutated_child=child
                t+=1
            
        return mutated_child
    
    def _getpopulationMatrix(self,total_columns):
        #generate chromosome based on number of features in base model and hyperparameter
        population_array=np.random.randint(0,2,total_columns)
        
        #shuffle after concatenating 0 abd 1
        np.random.shuffle(population_array)
        
        #create blank population matrix to append all individual chrososomes
        
        population_matrix=np.empty((0,total_columns))
        
        #get population matrix
        population_matrix=self._get_population(population_matrix,population_array)
        
        #best solution for each generation
        best_of_a_generation = np.empty((0,len(population_array)+1))
        
        return population_array,population_matrix,best_of_a_generation
    
    def GetBestFeatures(self):
        #record time
        start=time.time()
        
        
        if 0 in self.data_dict.keys():
            total_columns=len(self.columns_list)

        ##get population array to begin
        population_array,population_matrix,best_of_a_generation=self._getpopulationMatrix(total_columns=total_columns)
            

        for genrtn in range(self.generations):
            #if time exceeds, break loop
            if (time.time()-start)//60>self.run_time:
                print('================= Run time exceeded allocated time. Producing best solution generated so far. =================')
                break
            
            #placeholder for saving new generation
            new_population = np.empty((0,len(population_array)))
            
            #placeholder for saving new generation
            new_population_with_obj_val = np.empty((0,len(population_array)+1))
            
            #placeholder for saving best solution for each generation
            sorted_best = np.empty((0,len(population_array)+1))


            #doing it half population size will mean getting matrix of population size equal to original matrix
            for family in range(int(self.population/2)):

                parent1=[]
                parent2=[]
                
                while len(parent1)==0 and len(parent2)==0:
                    parent1,parent2=self._get_parents(population_array=population_array,population_matrix=population_matrix)

                #crossover
                child1=[]
                child2=[]
                while len(child1)==0 and len(child2)==0:
                    child1,child2=self._crossover(parent1=parent1,parent2=parent2)

                #mutation
                mutated_child1 = []
                mutated_child2 = []
                while len(mutated_child1)==0 and len(mutated_child2)==0:
                    mutated_child1=self._mutation(child=child1)
                    mutated_child2=self._mutation(child=child2)
                

                #get cost function for 2 mutated child and print for generation, family and child                
                fold_cost1=self._getCost(population_array=mutated_child1)
                fold_cost2=self._getCost(population_array=mutated_child2)
                
                #create population for next generation
                new_population=np.vstack((new_population,mutated_child1,mutated_child2))
                
                #save cost and child
                mutant1_with_obj_val=np.hstack((fold_cost1,mutated_child1))
                mutant2_with_obj_val=np.hstack((fold_cost2,mutated_child2))
                
                #stack both chromosome of the family
                new_population_with_obj_val = np.vstack((new_population_with_obj_val,mutant1_with_obj_val,mutant2_with_obj_val))
                
            #at end of generation, change population as the stacked chromosome set from previous generation
            population_matrix = new_population
            
            if self.cost_function_improvement=='increase':
                #find the best solution for generation based on objective function and stack
                sorted_best = np.array(sorted(new_population_with_obj_val,key=lambda x:x[0],reverse=True))
            elif self.cost_function_improvement=='decrease':
                #find the best solution for generation based on objective function and stack
                sorted_best = np.array(sorted(new_population_with_obj_val,key=lambda x:x[0],reverse=False))
            
            if self.print_iteration_result:
                print('Best combined performance on test and validation data for generation',genrtn,':',sorted_best[0][0])
            best_of_a_generation=np.vstack((best_of_a_generation,sorted_best[0]))

        if self.cost_function_improvement=='increase':
            #sort by metric
            best_metric_chromosome_pair=np.array(sorted(best_of_a_generation,key=lambda x:x[0],reverse=True))[0]
        elif self.cost_function_improvement=='decrease':
            best_metric_chromosome_pair=np.array(sorted(best_of_a_generation,key=lambda x:x[0],reverse=False))[0]

        #best chromosome, metric and vocabulary
        best_chromosome = best_metric_chromosome_pair[1:]

        columns_list=list(map(list(self.columns_list).__getitem__,self._get_feature_index(best_chromosome)))

        print('================= Best result:',best_metric_chromosome_pair[0],'=================')
        print('================= Execution time in minutes:',(time.time()-start)//60,'=================')
        return columns_list
    
    
class FeatureSelection:
    
    '''
    Machine Learning Parameters
    ----------

    columns_list : Column names present in x_train_dataframe and x_test_dataframe which will be used as input list for searching best list of features.

    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}
        
        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.
    
    x_validation_dataframe : dataframe containing features of validatoin dataset
    
    y_validation_dataframe : dataframe containing only the dependent variable of validation dataset
    
    model : Model object. It should have .fit and .predict attribute
        
    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'

    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.
    
    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class
        
    '''




    def __init__(self,columns_list,data_dict,x_validation_dataframe,y_validation_dataframe,model,cost_function,cost_function_improvement='increase',average=None):
        self.columns_list=columns_list
        self.data_dict=data_dict
        self.x_validation_dataframe=x_validation_dataframe
        self.y_validation_dataframe=y_validation_dataframe
        self.model=model
        self.cost_function=cost_function
        self.cost_function_improvement=cost_function_improvement
        self.average=average

    def _sanityCheck(self):
        if len(self.columns_list) == 0:
            print('Input field "columns_list" is empty. Cannot perform feature selection.')
            return False
        
        if self.data_dict[0]['x_train'].shape[0] != self.data_dict[0]['y_train'].shape[0]:
            print('Number of rows in "x_train" and "y_train" are not same. Cannot perform feature selection.')
            return False

        if self.x_validation_dataframe.shape[0] != self.y_validation_dataframe.shape[0]:
            print('Number of rows in "x_validation_dataframe" and "y_validation_dataframe" are not same. Cannot perform feature selection.')
            return False

        if self.x_validation_dataframe.shape[0] == 0:
            print('Number of rows in "x_validation_dataframe" is 0. Cannot perform feature selection.')
            return False

        if self.y_validation_dataframe.shape[0] == 0:
            print('Number of rows in "y_validation_dataframe" is 0. Cannot perform feature selection.')
            return False
        
        return True
        
        
    def GeneticAlgorithm(self,generations=100,population=50,prob_crossover=0.9,prob_mutation=0.1,run_time=120,print_iteration_result=True):
        '''
        Genetic Algorithm Parameters
        ----------        
        generations : Number of generations to run genetic algorithm. 100 as deafult
        
        population : Number of individual chromosomes. 50 as default. It should be kept as low number if number of possible permutation and combination of feature sets are small.
        
        prob_crossover : Probability of crossover. 0.9 as default
        
        prob_mutation : Probability of mutation. 0.1 as default
            
        run_time : Number of minutes to run the algorithm. This is checked in between generations.
            At start of each generation it is checked if runtime has exceeded than alloted time.
            If case run time did exceeds provided limit, best result from generations executed so far is given as output.
            Default is 2 hours. i.e. 120 minutes.

        print_iteration_result : Whether the user wishes to print result at each generation. Values are True/False. Default is True

        Output
        ----------
        best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
        
        '''
        
        if self._sanityCheck():
        
            gaOBJ = GeneticAlgorithmFS(
                columns_list=self.columns_list,
                data_dict=self.data_dict,
                x_validation_dataframe=self.x_validation_dataframe,
                y_validation_dataframe=self.y_validation_dataframe,
                model=self.model,
                cost_function=self.cost_function,
                cost_function_improvement=self.cost_function_improvement,
                average=self.average,
                generations=generations,
                population=population,
                prob_crossover=prob_crossover,
                prob_mutation=prob_mutation,
                run_time=run_time,
                print_iteration_result=print_iteration_result)
            
            best_columns = gaOBJ.GetBestFeatures()

            return best_columns
    
    def SimulatedAnnealing(self,temperature=1500,iterations=100,n_perturb=10,n_features_percent_perturb=1,alpha=0.9,run_time=120,print_iteration_result=True):
        '''
        Simulated Annealing Parameters
        ----------
        
        temperature : Initial temperature for annealing. Default is 1500
        
        iterations : Number of times simulated annealing will search for solutions. Default is 100
        
        n_perturb : Number of times feature set will be perturbed in an iteration. Default is 10.
    
        n_features_percent_perturb : Percentage of features that will be perturbed during each perturbation. Value are between 1 and 100. Default is 1.
        
        alpha : Temperature reduction factor. Defaults is 0.9
    
        run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
            At start of each generation it is checked if runtime has exceeded than alloted time.
            If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
            Default is 2 hours. i.e. 120 minutes.

        print_iteration_result : Whether the user wishes to print result at each iteration. Values are True/False. Default is True
    
        Output
        ----------
        best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
        
        '''
        if self._sanityCheck():
        
            saObj = SimulatedAnnealingFS(
                columns_list=self.columns_list,
                data_dict=self.data_dict,
                x_validation_dataframe=self.x_validation_dataframe,
                y_validation_dataframe=self.y_validation_dataframe,
                model=self.model,
                cost_function=self.cost_function,
                cost_function_improvement=self.cost_function_improvement,
                average=self.average,
                temperature = temperature,
                iterations = iterations,
                n_perturb = n_perturb,
                n_features_percent_perturb = n_features_percent_perturb,
                alpha = alpha,
                run_time = run_time,
                print_iteration_result=print_iteration_result)

            best_columns = saObj.GetBestFeatures()

            return best_columns

    
    def AntColonyOptimization(self,iterations=100,N_ants=100,evaporation_rate=0.9,Q=0.2,run_time=120,print_iteration_result=True):
        '''
        Ant Colony Optimization Parameters
        ----------
        
        iterations : Number of times ant colony optimization will search for solutions. Default is 100
        
        N_ants : Number of ants in each iteration. Default is 100.
    
        evaporation_rate : Evaporation rate. Values are between 0 and 1. If it is too large, chances are higher to find global optima, but computationally expensive. If it is low, chances of finding global optima are less. Default is kept as 0.9
        
        Q : Pheromene update coefficient. Value between 0 and 1. It affects the convergence speed. If it is large, ACO will get stuck at local optima. Default is kept as 0.2
    
        run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
            At start of each generation it is checked if runtime has exceeded than alloted time.
            If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
            Default is 2 hours. i.e. 120 minutes.

        print_iteration_result : Whether the user wishes to print result at each iteration. Values are True/False. Default is True

        Output
        ----------
        best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
            
        '''
        if self._sanityCheck():
        
            acOBJ = AntColonyOptimizationFS(
                columns_list=self.columns_list,
                data_dict=self.data_dict,
                x_validation_dataframe=self.x_validation_dataframe,
                y_validation_dataframe=self.y_validation_dataframe,
                model=self.model,
                cost_function=self.cost_function,
                cost_function_improvement=self.cost_function_improvement,
                average=self.average,
                iterations = iterations,
                N_ants = N_ants,
                evaporation_rate = evaporation_rate,
                Q = Q,
                run_time = run_time,
                print_iteration_result=print_iteration_result)

            best_columns = acOBJ.GetBestFeatures()

            return best_columns


    
    def ParticleSwarmOptimization(self,iterations=50,swarmSize=100,run_time=120,print_iteration_result=True):
        '''
        Particle Swarm Optimization Parameters
        ----------
        
        iterations : Number of times particle swarm optimization will search for solutions. Default is 100
        
        swarmSize : Size of the swarm in each iteration. Default is 100.
        
        run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
            At start of each generation it is checked if runtime has exceeded than alloted time.
            If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
            Default is 2 hours. i.e. 120 minutes.
        
        print_iteration_result : Whether the user wishes to print result at each iteration. Values are True/False. Default is True
        
        Output
        ----------
        best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.
            
        '''
        if self._sanityCheck():
        
            psOBJ = ParticleSwarmOptimizationFS(
                columns_list=self.columns_list,
                data_dict=self.data_dict,
                x_validation_dataframe=self.x_validation_dataframe,
                y_validation_dataframe=self.y_validation_dataframe,
                model=self.model,
                cost_function=self.cost_function,
                cost_function_improvement=self.cost_function_improvement,
                average=self.average,
                iterations = iterations,
                swarmSize = swarmSize,
                run_time = run_time,
                print_iteration_result=print_iteration_result)

            best_columns = psOBJ.GetBestFeatures()

            return best_columns

        