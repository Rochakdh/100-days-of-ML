![dataset] (https://github.com/Rochakdh/100-days-of-ML/blob/master/t-sne/MNIST-single-data.png?raw=true "fig :single_MNIST data set")

![t-sne] (https://github.com/Rochakdh/100-days-of-ML/blob/master/t-sne/t-sne_visualization.png?raw=true "t-SNE visualization of MNIST dataset")

Things learned :
1. t-sne is not a classifier or any algorithm to make clusters
2. t-sne is a visualizer of high dimensional data.
3. Generally it converts hyper dimension to 2-D for better visualization of data
4. t-sne converts distances from each neighbour to probabilistic values.
5. if they fall under a normal t-region, they are pulled together. If not the points are pushed
6. In this project even though a minst is a supervised classification problem, we visualized data in an unsupervised way.