eig = np.zeros([Nw, nG], dtype=complex)
eig_all = np.zeros([Nw, nG], dtype=complex)
weight = np.zeros([Nw,nG],dtype=complex) 

e_GG = e_wGG[0]
eig_all[0], vec = np.linalg.eig(e_GG)
eig[0] = eig_all[0]
vec_dual = np.linalg.inv(vec)

for i in np.array(range(1, Nw)):	
e_GG = e_wGG[i] 
eig_all[i], vec_p = np.linalg.eig(e_GG)  #valeur propre et vecteurs propres "droits"

vec_dual_p = np.linalg.inv(vec_p)  # vecteurs propres "gauche"

# Pas sur de � quoi ca sert
overlap = np.abs(np.dot(vec_dual, vec_p))
index = list(np.argsort(overlap)[:, -1])
if len(np.unique(index)) < nG:  # add missing indices
    addlist = []
    removelist = []
    for j in range(nG):
        if index.count(j) < 1:
            addlist.append(j)
        if index.count(j) > 1:
            for l in range(1, index.count(j)):
                removelist+= \
                    list( np.argwhere(np.array(index) == j)[l])
    for j in range(len(addlist)):
        index[removelist[j]] = addlist[j]
vec = vec_p[:, index]
vec_dual = vec_dual_p[index, :]
eig[i] = eig_all[i, index]
##	
        


weight[i]=vec[0,:]*(np.transpose(vec_dual[:,0])) #le poids de chaque mode est le produit scalaire entre les vecteurs propres pour G=0
