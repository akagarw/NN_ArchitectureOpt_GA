# Neural Network Architecture Optimization using Genetic Algorithms
  Given Python Code in "NN_ArchitectureOpt_GA.ipynb" Jupyter Notebook aims to solve the problem of Architecture Optimization in Neural Networks using Genetic Algorithms.  
  Here the Model is evaluated on the [Pima Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) Dataset and the end results are compared to the values obtained by the Authors of the MLP-LOA Research Paper \[[Springer](https://link.springer.com/article/10.1007/s00500-019-03773-2) | [SciHub](https://sci-hub.se/https://link.springer.com/article/10.1007/s00500-019-03773-2)\], which aims to solve the problem by various classifiers, one of which is GA. NN Models are created and trained using the TensorFlow Keras Libraries for convenience and execution speed up by making use of GPU resources.
  
### Overview:  
  The GA optimizers the Architecture of NN based on the NN Hyperparameters like - Learning Rate(alpha), Type of Activation Function, No. of Hidden Layers, No. of Hidden Neurons in each Hidden Layer. Represents each possible solution(Chromosome) using Binary Encoding in the genotype representation. Each such solution is converted to Phenotype representation(Binary to Decimal Base Conversion), Values are checked to be in the valid range and then a Keras Model is created and trained under TensorFlow library for Max 30 Epochs with Early Stopping. The Metrics are logged in and the Population is evolved throughout Generations by the help of Elitism, Selection, Crossover and Mutation Operators. The Metrics are plotted in the end across generations and the best Model(NN_ArchOpt_GA_bestModel_ckpt) is saved.
  

### NN Architecture Details:
- Training on Diabetes Dataset With Train/Val/Test Split : 0.6/0.1/0.3 
- Parameter details:

    | Parameter        	    |    Values/Range                       	    | No.of Bits Required |
    |-------------------    |-----------------------------------------------|---------------------|
    | Alpha            	    |     [ 0.0039 - 0.9961 ]               	    |          08         |
    | Activation Func  	    |    { Sigmoid - 0, Tanh - 1, ReLU - 2} 	    |          02         |
    | ctr(HL1 Neurons) 	    |    {5...31}                           	    |          05         |
    | ctr(HL2 Neurons) 	    |    {0...31}                           	    |          05         |
    
### Genetic Algorithm Parameters:
- Population Size = 50
- Max Generations = 30
- Using Binary Encoding to represent chromosome solution(Genotype Representation)
- Each Chromosome defined as = ( alpha | Activation Func | ctr(HL1 Neurons) | ctr(HL2 Neurons) )
- Total Genes = Total Bits = 20 
- Probability Of Crossover = 0.8
- Probability Of Mutation = 0.1
- Objective Function = NN Model Test Split Evaluation Loss and Accuracy
- Elitism = 10% of Population
- Selection Operator = Roulette Wheel Selection
- Crossover Operator = One Point Crossover
- Mutation Operator = One Point Mutation


## Algorithm Workflow

### Algorithm Setup:
  - Required Libraries are imported, the GA operating variables discussed above are declared and set to the required values in the Jupyter Kernel Environment
  - Diabetes dataset is loaded, split into Train/Test/Val Split and then standarized. 
  - Helper Functions for defining Chromosome Objects - Genotype to Phenotype Cnversions, binary to Decimal Conversion, Parameter Range Validity Check. 
  - Defining Chromosome Class : Encapsulating all Parameter Initialization in Genotype Form and phenotype conversion using Helper Functions, Keras Model Creation and Training followed by storing the Model Metrics.
  - Defining Generation Class : Encapsulating all Generation Operations to create Next Gen, Sets initial Gen, and declares metric storing variables, defining functions for Generation Metrics logging, elitism, roulette Wheel selection.
  
### Initial Population Generation:
  A Set of Chromosome Class Objects is generated. Each Chromosome Object is generated with the set of parameters, whose values are randomly generated in the binary representation, converted to real values and checked if within the range. Created the Keras Model, Trained the Model with Early Stopping and logged the Best Model Metrics as a Dictionary as Object's Attribute.
  
### Creating Next Generation:
  Generation Class Object is declared and initialized with Initial Generation List.
#### Fitness Calculation : 
The Logged Test Split Evaluation Loss and Accuracy of Each chromosome is by the objective function to calculate its Fitness. These Metrics are grouped together in the Generation Object's Gen Metrics Attributes and then the Fitness is calulated and logged to Gen Fitness attribute. Here all the Statistical Representations(Min, Max, Mean) of the generated metrics are stored for each generation.
#### Elitism:
Now, The Best 10% of the Population are chosen and taken as it is in the next Generation to keep best solution and control diversity and maintain exploration.
#### Selection:
To get the remaining population, from the Current Generation offsprings are generated. So based on the Fitness Metric, Two parents are selected at random 
by the Roulette Wheel Selection Method, from which Two offspring will be created based on their genetic material in genotype representation. 
#### Crossover and Mutation:
Based on a randomly generated number value, it is decided whether crossover and mutation operators will be performed on the Two chosen parents to generate corresponding offsprings. The parameters of generated off springs are checked if they are within the range else reset using the Helper Functions.
The Probability of Crossover and Probability of Mutation are kept as such to explore more in the initial iterations of creating new generations to increase diversity 
whereas do more of exploitation in the generations to decrease diversity and converge to the optimal solution found.

### Metrics Plotting
After iterating for max generations, the evalutation metrics(Loss, Accuracy and Fitness) are plotted accross generations.

## End Results:
  The Best Model obtained after execution of the Algorithm is saved in "NN_ArchOpt_GA_bestModel_ckpt" Folder. The Best Solution HyperParameters are : 
    - learning_Rate = 0.390625
    - Activation Function = ReLu
    - No. of Hidden Layers =2
    - Ctr(HL1 Neurons)=21
    - Ctr(HL2 Neurons)=30
    
  With the Best Model's Performance Metrics:
    - train_loss: 0.4218
    - train_accuracy: 0.7935
    - val_loss: 0.7926
    - val_accuracy: 0.6364
    - test_loss: 0.4306
    - test_accuracy: 0.7792
  
  The results may vary for executions at different times due to inculcation of stochastic process like random initialization in GA. 
 
 ## Comparison of Results:
  The Best Solution/Model Metrics after Iterating for Max Generation in the end, after Multiple Executions of the Algorithm are compared with the ones author have tabulated in the mentioned research paper. Looking over the obtained different results. The Best Model Accuracy was seen to be approximately equal to the findings in the research paper.  
  
  Hence validates the correctness of the operations of the written Algorithm from Scratch.
  

