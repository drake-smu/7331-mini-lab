# Estimating C parameter
X = X_train.append(X_test)
y = y_train.append(y_test)

X2 = X_train2.append(X_test2)
y2 = y_train2.append(y_test2)

svc = svm.SVC(kernel = 'linear')
C_s = np.logspace(-10, 0, 10)
#%%

def crossvalidator(v1,v2):
    scores = list() #cross vall scores
    scorestd = list() # Their SD
    for C in C_s:
        svc.C = C
        current_score = cross_val_score(svc, v1, v2, cv=5, n_jobs=12)
        scores.append(np.mean(current_score))
        scorestd.append(np.std(current_score))
        return {'scores': scores, 'scorestd': scorestd}
    
cvres = crossvalidator(X,y)
#%%

scores = cvres['scores']
scorstd = cvres['scorestd']
plt.figure()
plt.semilogx(C_s,scores)
#%%
plt.semilogx(C_s,np.array(scores) + np.array(scorestd), 'b--')
plt.semilogx(C_s,np.array(scores) - np.array(scorestd), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Cparam')
plt.ylim(0,1.1)
plt.show()