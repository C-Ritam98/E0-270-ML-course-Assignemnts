from sys import exec_prefix
import numpy as np
import matplotlib.pyplot as plt

class Expectation_maximisation:

    def __init__(self,name,data,n_class):
        self.name = name
        print(self.name,end=' ')
        print(f"Initiated with {n_class} classes!!")

        self.n_class = n_class
        np.random.seed(1)

        self.lamda = np.array([np.random.rand() for i in range(n_class)]).reshape(1,n_class) 
        self.tau = np.array([(1/n_class) for _ in range(n_class)]).reshape(1,n_class)

        self.gamma_ = np.array([[self.distribution(data[j,0],self.lamda[0,i]) for i in range(n_class)] for j in range(data.shape[0])])

        self.data = data

    def train(self,iter_num = 1000,to_plot = True):

        iter = 0
        lambda_1 = []
        lambda_2 = []
        tau_1 = []
        tau_2 = []

        while True:

            # E-step            
            self.gamma_ =  np.exp(-1 * self.data * self.lamda) * self.lamda
            self.gamma_ = self.gamma_ * self.tau
            self.gamma_ = self.gamma_/np.sum(self.gamma_,axis=1,keepdims=True)

            # M-step

            self.tau = np.sum(self.gamma_,axis = 0)/self.data.shape[0]
            self.lamda = 1/((self.gamma_.T @ self.data).T / np.sum(self.gamma_,axis=0,keepdims=True)) 
            #print(np.sum(gamma_ @ np.ones((2,1))) == data.shape[0])
            lambda_1.append(self.lamda[0,0])
            lambda_2.append(self.lamda[0,1])
            tau_1.append(self.tau[0])
            tau_2.append(self.tau[1])


            iter+=1

            if iter == iter_num:
                break
            

        output = np.argmax(self.gamma_,axis=1)
        #print(output)
        cell = dict()

        for i in range(self.n_class):
            cell[i] = []

        for i in range(self.data.shape[0]):
            cell[output[i]].append(self.data[i,0])

        ## plots
        plt.plot(lambda_1,label = "lambda_1")
        plt.plot(lambda_2, label = 'lambda_2')
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Lambda values")
        plt.title("Plot of lambda values with iteration")
        plt.show()
        plt.clf()

        plt.plot(tau_1, label = "tau_1")
        plt.plot(tau_2, label = 'tau_2')
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Tau values")
        plt.title("Plot of tau values with iteration")
        plt.show()
        plt.clf()

        if to_plot == True:
            self.plotting(cell)


    def plotting(self,cell):

        for i in range(self.n_class):
            print(f"Number of points in class {i+1} : {len(cell[i])}")
            plt.scatter(cell[i],[0]*len(cell[i]))
        plt.title("Segregated Data")
        plt.show()
            
            
    def distribution(self,x,lamda):
        return lamda* np.exp(-lamda * x)

if __name__ == "__main__":

    data = []

    with open("mixture_data.txt",'r') as f:
        lines = f.readlines()
        data = np.array([float(x) for x in lines]).reshape(-1,1)/100


    n_class = 2


    EM_for_exp = Expectation_maximisation("EM Class for exponential data ",data,n_class)
    EM_for_exp.train(10000,True)









