from tkinter import Y
import numpy as np
def normalize_add_one(X):
    X=np.array(X)
    X_max=np.max(X,axis=0)*np.ones(X.shape)
    X_min=np.min(X,axis=0)*np.ones(X.shape)
    X_normalized=(X-X_min)/(X_max-X_min)
    ones=np.ones([X.shape[0],1])
    return np.column_stack((ones,X_normalized))
class RidgeRegression:
    def __init__(self):
        return
    def fit(self,X_train,Y_train,Lambda):
        W=np.linalg.inv(X_train.transpose().dot(X_train)+Lambda*np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return W
    def fit_gradient(self,X_train,Y_train,Lambda):
        return 0
    def predict(self,W,X_new):
        X_new=np.array(X_new)
        Y_new=X_new.dot(W)
        return Y_new
    def compute_RSS(self,Y_new,Y_predicted):
        diff=np.sum((Y_new-Y_predicted)**2)
        loss=diff/Y_new.shape[0] 
        return loss
    def best_lambda(self,X_train,Y_train):
        def cross_validation(num_fold,LAMBDA):
            row_id = np.array(range(X_train.shape[0]))
            valid_id=np.split(row_id[:len(row_id)-len(row_id)%num_fold],num_fold)
            valid_id[-1]=np.append(valid_id[-1],row_id[len(row_id)-len(row_id)%num_fold:])
            rss=0
            for k in range(num_fold):
                train_id=[i for i in row_id if i not in valid_id[k]]
                train_set = {"X":X_train[train_id],"Y":Y_train[train_id]}
                valid_set = {"X":X_train[valid_id[k]],"Y":Y_train[valid_id[k]]}
                W0=self.fit(X_train=train_set['X'],Y_train=train_set['Y'],Lambda=LAMBDA)
                Y_predicted=self.predict(W0,valid_set['X'])
                rss += self.compute_RSS(valid_set['Y'],Y_predicted)
            return rss/num_fold
        def range_scan(best_lambda,min_rss,lambda_values):
            for i in lambda_values:
                current_rss=cross_validation(num_fold=5,LAMBDA=i)
                if current_rss < min_rss:
                    best_lambda=i
                    min_rss=current_rss
                return best_lambda,min_rss
        best_lambda,min_rss= range_scan(best_lambda=0,min_rss=1e6,lambda_values=range(30))
        lambda_values=[(k/1000) for k in range(max(0,best_lambda*1000,30))]
        best_lambda,min_rss=range_scan(best_lambda=best_lambda,min_rss=min_rss,lambda_values=lambda_values)
        return best_lambda
if __name__=='__main__':
    with open('D:\deathrate.txt') as f:
        X=np.loadtxt(f)
    Y=X[:,-1]
    X_train,Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]
    RR=RidgeRegression()
    best_lamdba= RR.best_lambda(X_train,Y_train)
    print("best lambda = ",best_lamdba)
    W_learned=RR.fit(X_train,Y_train,best_lamdba)
    Y_pred=RR.predict(W_learned,X_test)
    print (RR.compute_RSS(Y_test,Y_pred))
