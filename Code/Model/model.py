#coding:utf-8
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from scipy import interp 
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from collections import defaultdict
def load_drug_cell():
	linenumber=1
	druglist=[]
	drug_cell={}
	for i in open("mat_ccle_drug_cellLineIC50less0.5.txt").readlines():
		if linenumber==1:
			linenumber+=1
			celllist=i.strip().split('\t')
			continue
		newi=i.strip().split('\t')
		druglist.append(newi[0])
		#print celllist[0]
		for j in range(0,len(newi)-1):
			drug_cell[newi[0]+"\t"+celllist[j]]=newi[j+1]
	return druglist,celllist,drug_cell

def load_drug(druglist):
	j=0
	drug_map={}
	#for i in open("DrugChemicalNetFeature\\drug_vector_d10.txt").readlines():
	for i in open("1matlab\\src\\feature\\CCLE_drugFeatureTarChem\\drug_vector_d15.txt").readlines():
		newi=i.strip().split('\t')
		drug_map[druglist[j]]=newi
		j+=1
	return drug_map

def load_cell(celllist):
	j=0
	#cell_map=defaultdict(lambda: [0,])
	cell_map={}
	for i in open("1matlab\\src\\feature\\CCLE_cellFeatureExp\\cellLine_vector_d10.txt").readlines():
		newi=i.strip().split('\t')
		cell_map[celllist[j]]=newi
		j+=1
	return cell_map

def write_data(druglist,celllist,drug_cell,drug_map,cell_map):
	
	
	for drug in druglist:
		fw=open("data\\data"+drug,"w")
		drug_vec=drug_map.get(drug) 
		for cell in celllist:
			cell_vec=cell_map.get(cell)
			#print(len(cell_map)) 
			label=drug_cell.get(drug+"\t"+cell)
			S=""
			for vec1 in drug_vec:
				S=S+vec1+","
			for vec2 in cell_vec:
				S=S+vec2+","
			S+=label+"\n"
			fw.write(S)
	fw.flush()
	fw.close()

def load_data():
	
	drug_files = os.listdir("data")
	ans = {}
	o = 1
	for drug in drug_files:
		#print o
		#o += 1
		rate = 0
		rate_0=0
		Feature=[]
		for i in open("data/"+drug,'r').readlines():
			X_feature=[]
			newi=i.strip().split(',')
			#print len(newi)
			for newnewi in newi:
				X_feature.append(float(newnewi))
			if newi[-1] == '1':
				rate += 1
			if newi[-1]== '0':
				rate_0+=1
			X_feature=np.array(X_feature)
			#print X_feature.shape
			Feature.append(X_feature)
			#print np.array(Feature).shape
		print(drug)
		print (rate_0)
		print(rate)
		rate = rate*1.0/(rate_0+rate)
			
		ans[drug] = np.array(Feature)
	return ans,rate
if __name__=="__main__":

	druglist,celllist,drug_cell=load_drug_cell()
	drug_map=load_drug(druglist)
	cell_map=load_cell(celllist)	 
	write_data(druglist,celllist,drug_cell,drug_map,cell_map)
	
	
	ans, rate = load_data()
	
	for t in ans.keys():
		print(t+":")
		X=ans[t][:,:-1]
		Y=ans[t][:,-1]
		index = np.random.permutation(X.shape[0])
		X=X[index,:]
		Y=Y[index]
		#print(X.shape)
		from sklearn.decomposition import PCA
		#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
		#pca= PCA(n_components=80)
		
		#pca.fit(X)
		#clf=LogisticRegression()
		#clf = RandomForestClassifier(n_estimators=400, max_depth=100,class_weight={0:0.1,1:0.9},n_jobs = 4)
		#clf = RandomForestClassifier(n_estimators=100, max_depth=100,class_weight={0:3,1:1000},n_jobs = 4)
		clf2=SVC(probability=True)
		#clf2=AdaBoostClassifier(n_estimators=3)
		clf1=GradientBoostingClassifier(n_estimators=50)
		# 0.05 : 0.995 1 
		#0.08: 0.92 1
		#0.1: 0.9 0
		#0.09:0.91 0
		#0:0.085,1:0.915 0
		#1:1000 0.1236 3:1000 0.1736 
		#clf.fit(X,Y)
		#y_pre_test= clf.predict_proba(X)
		'''
		fw=open("result","w")
		fw.write("true,predice\n")
		y_pre_test= clf.predict(X)
		for i in range(0,len(y_pre_test)):
			fw.write(str(Y[i])+","+str(y_pre_test[i])+"\n")
		print "ppppp"
		fw.flush()
		fw.close()
		'''
		'''
		mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)  
		all_tpr = []
		#loo = LeaveOneOut()
		cv = StratifiedKFold(Y, n_folds=80)
		for i, (train, test) in enumerate(cv):
			#probas_ = clf.fit(X[train], Y[train]).predict_proba(X[test])
			predict = clf.fit(X[train], Y[train]).predict(X[test])
			#fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
			fpr, tpr, thresholds = roc_curve(Y[test], predict)
			mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
			mean_tpr[0] = 0.0                               #初始处为0
			roc_auc = auc(fpr, tpr)  
			#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
			plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  
			#画对角线  
			plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
			mean_tpr /= len(cv)                     #在mean_fpr100个点，每个点处插值插值多次取平均  
			mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
			mean_auc = auc(mean_fpr, mean_tpr)      #计算平均AUC值  
			#画平均ROC曲线
			#print mean_fpr,len(mean_fpr)  
			#print mean_tpr  
			plt.plot(mean_fpr, mean_tpr, 'k--',  
					 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
			plt.xlim([-0.05, 1.05])  
			plt.ylim([-0.05, 1.05])  
			plt.xlabel('False Positive Rate')  
			plt.ylabel('True Positive Rate')  
			plt.title('Receiver operating characteristic example')  
			plt.legend(loc="lower right")  
			plt.show()  
		#print  y_pre_test[:,1]
		'''
		#loocv = LeaveOneOut()
		clf3 = LogisticRegression()
		eclf = VotingClassifier(estimators=[('GradientBoostingClassifier', clf1), ('svm', clf2), ('LogisticRegression', clf3)],weights=[1,1,1],voting='soft')
		fw=open("result\\ccle_IC50less0.5\\"+"ccle_ExpTarChem_c10d15.txt","a")
		for clf, label in zip([clf1, clf2, clf3, eclf], ['GradientBoostingClassifier', 'svm', 'LogisticRegression', 'Ensemble']):
			scores = cross_val_score(clf, X, Y, cv=5, scoring='roc_auc')
			print("roc_auc: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
			fw.write(t)
			fw.write("\t")
			fw.write(label)
			fw.write("\t") 
			fw.write(str(scores.mean()))
			fw.write("\t")
			fw.write(str(scores.std()))
			fw.write("\n")
			fw.flush()
		fw.close()
		#scores1 = cross_val_score(clf3, X, Y, cv=5,n_jobs=3,scoring='roc_auc')#,scoring='roc_auc'
		
		#print "\n"
		#print scores1
		#print "auc value: %0.4f (+/- %0.4f)" % (scores1.mean(), scores1.std() * 2)
		'''
		scores2 = cross_val_score(clf, X, Y, cv=10,scoring='recall',n_jobs=3)#,scoring='f1_macro'
		print scores2
		print "Recall: %0.4f (+/- %0.4f)" % (scores2.mean(), scores2.std() * 2)
		scores3 = cross_val_score(clf, X, Y, cv=10,scoring='f1',n_jobs=3)#,scoring='f1_macro'
		print scores3
		print "F1 score: %0.4f (+/- %0.4f)" % (scores3.mean(), scores3.std() * 2)
		'''
       