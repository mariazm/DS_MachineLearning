import pandas as pd
import numpy as np
import collections
from sklearn import decomposition
from sklearn import cross_validation
from sklearn import preprocessing

def initial_format(data):
	#Replacing with NAN's
	data = data.replace(r'\s+', np.nan, regex=True)

	# drop the column if it has all NaN values #From 1903 to 1874 No need if it is the inner
	data = data.dropna(axis='columns', how='all')
	#data = data[pd.notnull(data['lagMonth2'])]
	return data

def var_cat_treatment(data):
	#Droping variables with flat val
	flat_vars = ['intethn', 'oversamp', 'formwt','vstrat']
	data.drop(flat_vars, axis =1, inplace=True)

	cat_vars = ['sex','vpsu','evwork','wrkslf','wrkgovt','divorce','widowed','spevwork','spwrkslf','pawrkslf','mawrkslf','race','mawork','mawkbaby','mawkborn','mawk16','mawrkgrw','born','spkath','colath','libath','spkrac','colrac','librac','spkcom','colcom','libcom','spkmil','colmil','libmil','spkhomo','colhomo','libhomo','cappun','gunlaw','wirtap','grass','postlife','prayer','racmar','raclive','racclos','racinteg','rachome','racschol','racfew','rachaf','racmost','busing','racpres','racchurh','partfull','drink','drunk','smoke','quitsmk','smokecig','evsmoke','richwork','wksub','wksubs','wksup','wksups','unemp','govaid','fehome','fework','fepol','pill','teenpill','sexeduc','porninf','pornmorl','pornrape','pornout','xmovie','letdie1','suicide1','suicide2','suicide3','suicide4','hit','gun','hitok','hitmarch','hitdrunk','hitchild','hitbeatr','hitrobbr','polhitok','polabuse','polmurdr','polescap','polattak','fear','burglr','robbry','racdif1','racdif2','racdif3','racdif4','draft','draftem','memfrat','memserv','memvet','mempolit','memunion','memsport','memyouth','memschl','memhobby','memgreek','memnat','memfarm','memlit','memprof','memchurh','memother','reborn','savesoul','kid5up','united','visitart','makeart','dance','gomusic','perform','seemovie','plymusic','relexp','drama','othlang','compuse','usewww','download','mustwork','secondwk','partteam','hvylift','handmove','rincblls','laidoff','wkageism','wkracism','wksexism','wkharsex','wkharoth','backpain','painarms','extrapay','obeylaw','verdict','malive','empself','smallbig','privgovt','relexper','obeythnk','pubdecid','busdecid','grngroup','grnsign','grnmoney','grndemo','matesex','frndsex','acqntsex','pikupsex','paidsex','othersex','evpaidsx','condom','relatsex','evidu','evcrack','worda','wordb','wordc','wordd','worde','wordf','wordg','wordh','wordi','wordj','rvisitor','uswar','uswary','usintl','usun','whoelse1','whoelse2','whoelse3','whoelse4','whoelse5','whoelse6','mode','intsex','ballot','issp','phase','mobile16','natspac','natenvir','natheal','natcity','natcrime','natdrug','nateduc','natrace','natarms','nataid','natfare','natroad','natsoc','natmass','natpark','natchld','natsci','natspacy','natenviy','nathealy','natcityy','natcrimy','natdrugy','nateducy','natracey','natarmsy','nataidy','natfarey','courts','fund','fund16','spfund','spfund16','racdin','racopen','blksimp','happy','hapmar','life','helpful','fair','trust','confinan','conbus','conclerg','coneduc','confed','conlabor','conpress','conmedic','contv','conjudge','consci','conlegis','conarmy','aged','jobfind','satfin','finalter','getahead','fepres','divlaw','pornlaw','hitage','hitnum','gunage','owngun','pistol','shotgun','rifle','rowngun','ticket','arrest','comprend','form','racchng','rushed','milpay','fenumok','hinumok','blnumok','discaff','alike1','alike2','alike3','alike5','hlthinfo','waypaid','wkpraise','wkbonus','jobfind1','trynewjb','workfor','wrknokid','wrkbaby','wrksch','wrkgrown','mawork14','naturgod','sexsex','sexsex5','evstray','genetst1','geneself','raceself','dwelown','feeused','vote76','vote80','vote84','vote88','vote92','vote96','vote00','tax','reliten','dejavu','esp','visions','spirits','grace','bible','racpush','racseg','racdis','affrmact','satjob','union','chldmore','pillok','premarsx','teensex','xmarsex','homosex','spanking','hunt','coop','fechld','fehelp','fepresch','fefam','divorce5','hosdis5','chlddth','sibdeath','spdeath','trarel1','trarel5','milqual','punsin','blkwhite','rotapple','permoral','trstprof','fejobaff','discaffm','discaffw','rellife','relpersn','sprtprsn','chngtme','famwkoff','wkvsfam','famvswk','learnnew','workfast','workdiff','lotofsay','wktopsat','overwork','knowwhat','myskills','respect','trustman','safetywk','safefrst','teamsafe','safehlth','proudemp','prodctiv','wksmooth','trdunion','wkdecide','setthngs','toofewwk','promteok','opdevel','hlpequip','haveinfo','wkfreedm','fringeok','supcares','condemnd','promtefr','cowrkint','jobsecok','suphelp','wrktime','cowrkhlp','trainops','satjob1','talkemp','donothng','difstand','othcredt','putdown','lackinfo','actupset','shout','treatres','lookaway','protest1','protest3','protest6','revspeak','revpub','databank','jobsall','pricecon','hlthcare','aidold','aidindus','aidunemp','equalize','aidcol','aidhouse','posslq','hapunhap','premars1','xmarsex1','homosex1','godchnge','afterlif','heaven','hell','miracles','cantrust','postmat1','postmat2','scitest2','scitest4','scitest5','grntest1','grntest3','grntest4','nomeat','askfinan','askcrime','askdrugs','askmentl','askforgn','askdrink','asksexor','ethnum','spethnum','racedbtf','vetyears','visitors','commun','easyget','marital','degree','padeg','madeg','spdeg','famdif16','incom16','granborn','srcbelt','pres76','pres84','pres88','pres92','pres96','pres00','neargod','sprel16','wrkwayup','manners','success','honest','clean','judgment','control','role','amicable','obeys','responsi','consider','interest','studious','obey','popular','thnkself','workhard','helpoth','joblose','jobinc','jobsec','jobhour','jobpromo','jobmeans','class','finrela','parsol','news','divrel1','padeath','madeath','trauma1','trauma5','helppoor','helpnot','helpsick','helpblk','sunsch16','gochurch','believe','follow','liveblks','livewhts','marblk','marasian','marhisp','marwht','selfirst','fehire','wrktype','manvsemp','fairearn','health1','usedup','ownstock','talksup','othshelp','careself','peoptrbl','selffrst','wkstress','poleff3','setwage','setprice','cutgovt','makejobs','lessreg','hlphitec','savejobs','cuthours','spenviro','sphlth','sppolice','spschool','sparms','spretire','spunemp','sparts','govtpow','polint','poleff11','poleff13','poleff15','poleff16','poleff17','goodlife','inequal1','inequal3','inequal5','inequal6','incgap','goveqinc','taxrich','taxmid','taxpoor','taxshare','mawrkwrm','kidsuffr','famsuffr','homekid','housewrk','fejobind','twoincs','hubbywrk','marhappy','marnomar','marlegit','marhomo','kidjoy','kidnofre','kidempty','hubbywk1','meovrwrk','singlpar','cohabok','cohabfst','divbest','twoincs1','timepdwk','timehhwk','timefam','timefrnd','timeleis','wrkearn','wrkenjoy','secjob','hiinc','promotn','intjob','wrkindp','hlpoths','hlpsoc','flexhrs','stress','bossemps','concong','conbiz','conchurh','concourt','conschls','clergvte','clerggov','churhpow','theism','fatalism','godmeans','nihilism','egomeans','privent','scifaith','harmgood','scigrn','grnecon','harmsgrn','grnprog','grwthelp','antests','grwtharm','grnprice','grntaxes','grnsol','toodifme','ihlpgrn','carsgen','carsfam','nukegen','indusgen','chemgen','watergen','tempgen','recycle','chemfree','drivless','comtype','letin1','suiknew','sectech','secdocs','knomentl','dwelngh','dwelcity','hhrace','saqissp','saqsex','res16','pres80','pray','kidssol','phone','divrel4','unrel1','unrel4','god','readword','racwork','wrksched','wrkhome','givblood','givhmlss','retchnge','cutahead','volchrty','givchrty','givseat','helpaway','carried','directns','loanitem','selfless','accptoth','helphwrk','lentto','talkedto','helpjob','fambudgt','povline','famgen','feelevel','version','sample','polviews','eqwlth','likediff','mindbody','palefull','mapa','mastersp','judgeluv','frndking','world1','world4','satcity','sathobby','satfam','satfrnd','sathealt','socrel','socommun','socfrend','socbar','tratot1','wlthwhts','wlthblks','wlthasns','wlthhsps','workwhts','workblks','workasns','workhsps','intlwhts','intlblks','webyr','localnum','marelkid','parelkid','religkid','feelrel','sexfreq','intrace1','wrkstat','spwrksta','region','partyid','socpars','socsibs','chldnum','inperson','byphone','letters','meetings','byemail','cideknew','aidsknow','incdef','rplace','inthisp','reg16','family16','parborn','unrelat','earnrs','xnorcsiz','attend','maattend','paattend','spattend','closeblk','closewht','chldidel','unemp5','death5','attend12','relig','relig16']

	#Last class is dropped : to avoid problems of linear comb (intro class), it will be found setting the rest as 0
	#usually nan category will be dropped, but i just pointed it as -1 in the values to make things easier
	result =  data.copy()
	for cat in cat_vars:
		test= data[[cat]]
		#Create dummies, includes the NA ones
		agg = pd.get_dummies(test[cat], prefix="%s" %(cat), dummy_na = True)
		#Dropping last class
		agg = agg.drop(agg.columns[-1] , axis=1)
		#contatenating by index
		result = pd.concat([result, agg], axis=1, join_axes=[test.index])
		#Deleting original cat var from dataset
		result.drop([cat], axis =1, inplace=True)
	return result


def var_dummy(result):
	##Note that CPS vars already have dummy vars, and original ones  were inputted 0 
	##(as the total of all cats must be less than 1)
	##Now, I create dummys for the rest of vars, no imputation is done in this case as it will be mean (just based on training data)
	for feature in result.columns:
		if result[feature].isnull().sum() > 0:
			newColName = feature + '_dummy'
			result[newColName] = np.where(result[feature].isnull(), 1, 0)
	return result

def train_test_split(result):
	#move target variable 'target' to the first column
	target = result['target']
	result.drop('target', axis=1, inplace=True)
	result.insert(0, 'target', target)

	result['proAbortionCaseDecision'] = np.where(result['panelvote'] >= 2, 1, 0)
	result.drop([ 'year_month'], axis=1,inplace =True)
	#Following Kristen's script

	n = result.shape[0]
	# The split variable contains shuffled indices for the training data and for the testing data
	split = cross_validation.ShuffleSplit(n, n_iter=1, train_size = 0.8, test_size=.20, random_state = 1)

	train_idx = np.arange(n)
	test_idx = np.arange(n)

	for tr, te in split:
		train_idx = set(tr)
		test_idx = set(te)

	train_f = result.iloc[list(train_idx), :]         # convert train_idx from array to list of indices
	test_f = result.iloc[list(test_idx), :]           # convert test_idx from array to list of indices
	return train_f, test_f

def imput_train_test_missing(train1, test1):
	#features = train1.columns

	# for feature in features:  
	# 	if (feature != "lagdate"):
	# 		if (train1[feature].dtype != "float"):
	# 			train1[feature] = train1[feature].astype(float)              
	#Getting train means
	MVI = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(train1)

	mv_test1=  pd.DataFrame(MVI.transform(test1))
	mv_train1 =  pd.DataFrame(MVI.transform(train1))

	mv_test1.columns = test1.columns
	mv_train1.columns = train1.columns

	return mv_train1, mv_test1

def train_test_std(train1, test1):
	std_Train1 = train1.copy(deep=True).reset_index(drop=True)
	std_Test1 = test1.copy(deep=True).reset_index(drop=True)


	for feature in std_Train1.columns[2:]:   #standardize all the features (not standardizing the target)
	    arrayReshapeTrain = (std_Train1[feature].reshape(-1,1))    # reshape from a dataframe to an array per sklearn specifications 
	    scaler = preprocessing.StandardScaler().fit(arrayReshapeTrain)   
	    std_Train1[feature] = pd.DataFrame(scaler.transform(arrayReshapeTrain))

	    # apply the same scaler from the training data for the test and validation data
	    arrayReshapeTest = (std_Test1[feature].reshape(-1,1))
	    std_Test1[feature] = pd.DataFrame(scaler.transform(arrayReshapeTest))
	    
	return std_Train1, std_Test1



if __name__ == '__main__':
	#Charging dataset 
	print "Task: Cleaning and splitting merged datasets"
	data =  pd.read_csv("../1.Inputs_Merge/2merged_gss_ucr.csv", low_memory=False)
	formatted = initial_format(data)
	print "Initial shape %s" %(str(data.shape))
	print "Creating categorica binnings"
	print "Processing...please wait!"
	categorical =  var_cat_treatment(formatted)
	print "Cat shape %s" %(str(categorical.shape))
	print "Creating missing dummies"
	final_merged =  var_dummy(categorical)
	print "MV shape %s" %(str(final_merged.shape))
	print "Splitting train test"
	train, test = train_test_split(final_merged)
	print "Train shape %s" %(str(train.shape))
	print "Test shape %s" %(str(test.shape))
	mv_train, mv_test = imput_train_test_missing(train, test)
	final_train, final_test =  train_test_std(mv_train, mv_test)
	print "Exporting pickles of final results"
	final_train.to_pickle('./2_train_rev_cat')
	final_test.to_pickle('./2_test_rev_cat')



