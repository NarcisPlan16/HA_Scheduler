# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import joblib
import holidays
import warnings
warnings.filterwarnings('ignore')


class forecaster:
        def __init__(self, debug = False):
            """
            Constructor per defecte
            """
            
            self.debug = debug  # per mostrar per consola prints!
            self.search_space_conf_file = '../search_space.conf'
            self.db = dict()  # El model que hem creat.
        
        def windowing_grup(self, datasetin, look_back_ini=24, look_back_fi=48):
            """
            Funcio per crear les variables del windowing. Treballa sobre un dataset i inclou la variable objectiu!
            les variables creades es diran com l'original (legacy) i s'afegira '_' i el numero de desplacament al final del nom.
            Es tindran en compte les hores en el rang [ini, fi)
            
            Parametres:
                datasetin - pandas dataframe amb datetime com a index
                
                look_back_ini - on comença la finestra numero d'observacions (24 -> el dia anterior si es horari)
                
                look_back_fi - fins on arriba numero d'observacions (48-< el dia anterior si es orari)
                

            Retorna:
                dataset - el datasetin + les variables desplaçades en columnes noves
            """
            
            dataset = datasetin.copy()
            for i in range(0, len(dataset.columns)):
                for j in range(look_back_ini, look_back_fi):
                    dataset[dataset.columns[i]+'_'+str(j)] = dataset[dataset.columns[i]].shift(j)
            
            return dataset
        
        def windowing_univariant(self, datasetin, look_back_ini=0, look_back_fi=24, variable=''):
            """
            Funcio per crear les variables del windowing. de la variable indicada Treballa sobre un dataset i inclou la variable objectiu!
            les variables creades es diran com l'original (legacy) i s'afegira '_' i el numero de desplacament al final del nom.
            Es tindran en compte les hores en el rang [ini, fi).
            
            Parametres:
                datasetin - pandas dataframe amb datetime com a index
                
                look_back_ini - on comença la finestra numero d'observacions (25 -> el dia anterior si es horari)
                
                look_back_fi - fins on arriba numero d'observacions (48-< el dia anterior si es horari)

                vari - la variable on apliquem el windowing
                
            Retorna:
                dataset - el datasetin + les variables desplaçades en columnes noves
            """
            dataset = datasetin.copy()
            for i in range(0, len(dataset.columns)):
                if dataset.columns[i] == variable:
                    for j in range(look_back_ini, look_back_fi-1):
                        dataset[dataset.columns[i]+'_'+str(j)] = dataset[dataset.columns[i]].shift(j)

            return dataset

        def do_windowing(self, data, look_back={-1:[25, 48]}):

            if look_back is not None:  # torna un np array no un list l'object storer

                # windowing de totes les no especificades individualment
                if -1 in look_back.keys():  # si indicador es -1 volen un grup
                    # volen fer un grup
                    aux = look_back[-1]  # recuperem els valors de la finestra per el grup
                
                    # recuperem les que es faran soles
                    keys = list()
                    for i in look_back.keys():
                        if i != -1:
                            keys.append(i)  # les anem posant a una llista totes les que tenim exepte el-1

                    dad = data.copy()  # copiem el dataset per no perdre les que niran soles
                    dad = dad.drop(columns=keys)  # eliminem les que van soles

                    # fem windowing de tot el grup
                    dad = self.windowing_grup(dad, aux[0], aux[1])

                    # afegim les que aviem tret
                    for i in keys:
                        dad[i] = data[i]

                else:
                    # cas de que no tinguem grups son totes individuals, ho preparem tot per fer les individuals
                    dad = data.copy()  # copiem el dataset
                    # les que es faran soles
                    keys = list()
                    for i in look_back.keys():
                        if i != -1:
                            keys.append(i)

                # windowing de la resta que es fan 1 a 1
                variables = [col for col in data.columns if col not in keys]
                for i in variables:
                    aux = look_back[-1]
                    dad = self.windowing_univariant(dad, aux[0], aux[1], i)

            else:
                # cas de no tenir windowing
                dad = data.copy()

            return dad
        
        def colinearity_remove(self, datasetin, y, level=0.9):
            """
            Elimina les coliniaritats entre les variables segons el nivell indicat.
            parametres:
                datasetin - pandas dataframe amb datetime com a index
                
                y- la variable objectiu (per comprovar que no ens la carreguem!)
                
                level - el percentatge de correlacio de pearson per carregar-nos variables. None per no fer res
                
            Retorna:
                dataset - El dataset - les variables eliminades 
                to_drop - les variables que ha eliminat
            
            """
            if level is None:
                dataset = datasetin
                to_drop = None
            else:
                # ens carreguem els atributs massa correlacionats (colinearity)
                corr_matrix = datasetin.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > level)]
                if np.array(to_drop == y).any():  # la classe sempre hi ha de ser!!! millor asegurem que hi sigui!
                    del to_drop[to_drop == y]
                datasetin.drop(to_drop, axis=1, inplace=True)
                dataset=datasetin.copy()
                
            return [dataset, to_drop]
        
        def Model(self, X, y, algorithm='RF', params=None, max_time=None):
            """
            Funcio que realitza un randomized search per trovar una bona configuracio de l'algorisme indicat, o directament es crea amb els parametres indicats
            
            X- np.array amb les dades
            y- np.array am les dades
            """
            ## primer carreguem el grid de parametres des de fitxer i imports
            import json
            with open(self.search_space_conf_file) as json_file:
                d = json.load(json_file)
             
            if params is None:
                #No tenim parametres els busquem. Utilitzarem del fitxer.

                #preparem les dades de train i test.
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#, shuffle=False) #no fem shufle. volem que aprengui tot el periode no nomes les ultimes observacions.
                
                #inicialitzem llibreries i MAE maxim trobat.
                from sklearn.model_selection import ParameterSampler
                from sklearn.metrics import mean_absolute_error
                import time
                best_mae = np.inf
                
                # preparem la llista d'algorismes que volem provar. (tots  o el que ens indiquen.)
                if algorithm is None:
                    # no ens passen res. els probem tots
                    algorithm_list = list(d.keys())
                    
                elif isinstance(algorithm, list):
                    # ens passen llista a probar
                    algorithm_list = algorithm
                    
                else:
                    # ens passen nomes 1
                    algorithm_list = [algorithm]
                
                # per cada algorisme a probar...
                for i in range(len(algorithm_list)):
                    
                    #Recuperem grid de parametres
                    random_grid = d[algorithm_list[i]][0]
                    
                    if max_time is None:
                        iters = d[algorithm_list[i]][1]
                    else:
                        iters = max_time
                    
                    #Els strings per fer l'import corresponent a l'algorisme
                    impo1 = d[algorithm_list[i]][2]
                    impo2 = d[algorithm_list[i]][3]

                    if self.debug:
                        print(" ")
                        print("Començant a optimitzar: " + algorithm_list[i] + '- Algorisme ' + str(algorithm_list.index(algorithm_list[i])+1) + ' de ' + str(len(algorithm_list)) + ' - Maxim comput aprox (segons): ' + str(iters))

                    # preparem mostra aleatroia de parametres
                    sampler = ParameterSampler(random_grid, n_iter=np.iinfo(np.int64).max)
                    
                    # recuperem la llibreria correcte
                    a = __import__(impo1, globals(), locals(), [impo2])
                    Forcast_algorithm = eval("a."+impo2)
                    
                    try:
                        #creem i evaluem els models 1 a 1
                        t = time.perf_counter()
                        if self.debug:
                            print("Maxim " + str(len(sampler))+ " configuracions a probar!")
                            j=0
                            
                        for params in sampler:
                            regr = Forcast_algorithm(**params)
                            pred_test = regr.fit(X_train, y_train).predict(X_test)
                            act = mean_absolute_error(y_test, pred_test)
                            if best_mae > act:
                                best_model = regr
                                best_mae = act

                            if self.debug:
                                print("\r", end="")
                                j=j+1
                                print(str(j) + "/"+  str(len(sampler)) +" Best MAE: " +str(best_mae) +" ("+ type(best_model).__name__ + ") Last MAE: " + str(act) + " Elapsed: "+ str(time.perf_counter() - t) +" s         ", end="")
                                
                            if (time.perf_counter() - t) > iters:
                                if self.debug:
                                    print("Algorisme " + algorithm_list[i] + '  -- Hem arribat a fi de temps de cerca.' )
                                break
                              
                    except Exception as e:
                        print("Warning: Algorisme " + algorithm_list[i] + '  -- Abortat Motiu:' + str(e) )
                
                best_model.fit(X, y)
                model = best_model
                score = best_mae
                
                return [model, score]
            
            else:
                # ens han especificat algorisme i parametres
                
                # importem la llibreria correcte
                try:
                    # Els strings per fer l'import corresponent a l'algorisme
                    impo1 = d[algorithm][2]
                    impo2 = d[algorithm][3]
                except:
                    raise ValueError('Undefined Forcasting Algorithm!')
                    
                #recuperem la llibreria correcte
                a = __import__(impo1,globals(), locals(),[impo2])
                Forcast_algorithm = eval("a."+impo2 )
                
                # posem els parametres que ens diuen i creem el model
                f=Forcast_algorithm()
                f.set_params(**params)
                f.fit(X,y)
                score = 'none'
                return [f, score]

        # Feature selection/reduction methods
        def treu_atrs(self,X,y, metode=None):
            """
            Fem una seleccio d'atributs
            
            X- np.array amb les dades
            y- np.array am les dades
            
            metode  -   None = no fa res
                        integer = selecciona el numero de features que indiquis
                        PCA = Aplica un PCA per comprimir el dataset.
            """
            if metode is None:
                model_select = []
                X_new = X
            elif metode == 'Tree':
                from sklearn.ensemble import ExtraTreesRegressor
                from sklearn.feature_selection import SelectFromModel
                clf = ExtraTreesRegressor(n_estimators=50)
                clf = clf.fit(X, y)
                model_select = SelectFromModel(clf, prefit=True)
                X_new = model_select.transform(X)
            elif type(metode) is int:                
                from sklearn.feature_selection import SelectKBest, f_classif
                model_select = SelectKBest(f_classif, k=metode)
                X_new = model_select.fit_transform(X, y)
            elif metode == 'PCA':
                from sklearn.decomposition import PCA
                model_select = PCA(n_components='mle')# Minka’s MLE is used to guess the dimension
                X_new = model_select.fit_transform(X)
            else:
                raise ValueError('Undefined atribute selection method!')
                
            return [model_select, X_new, y]

        """
        A partir d'aqui tenim les 2 funcions que controlen tot el funcionament del forcasting (create_model - crear i guardar el model, i forcasting - recuperar i utilitzar el model)
        """
        
        def create_model(self, data, y, look_back={-1:[25,48]}, extra_vars={'variables':['Dia','Hora','Mes'], 'festius':['ES','CT']},
                         colinearity_remove_level=0.9, feature_selection='Tree', algorithm='RF', params=None, escalat=None, max_time=None):
            """
            Funcio per crear, guardar i configurar el model de forcasting.
                datasetin - pandas dataframe amb datetime com a index, format de sempre...
                
                y - nom de la columne amb la variable objectiu
                
                look_back - Windowing a aplicar, None = no es fara, altrament es un diccionari on la clau es la variable a fer windowing i el valor la finestra que se li ha d'aplicar.
                            Les claus, son strings indicant el nom de la columna a aplicar el windowing, si com a clau es dona -1 la finestra aplicara a totes les variables no especificades individualment.
                            
                            Els valors, son els que defineixen la finestra a aplicar i poden ser [ini, fi] o be [ini, fi, salt]  siguent on comença i acava la finestra de windowing en numero d'observacions (25 -> inici del dia anterior si es horari),  fi es fins on arriba numero d'observacions (48-> ultima hora del dia anterior si es orari), salt es per si no es vol una finestra continua i es vol saltar 24 observaxons per exemple. Amb None no fara res. ULL a no incloure coses que no tindrem en el moment d'execucio!!!!
                            exemple:
                                {'y':[25,48],  #la variable y de 25 a 48 observacions anteriors
                                 -1 : [1,480,24] #totes les que no siguin la 'y' de 1 a 480 pero amb salts de 24 es a dir la 1,25,49,...
                                }
                
                extra_vars - Per crear variables noves. Si es None no creara res.
                            espera un diccionari on les claus poden ser 'variables' per les generades des de l'index o be 'festius'
                            exemple: {'variables':['dia','hora','mes'],'festius':['ES','CT']}
                            A variables s'indicara la llista de variables de temps a posar com a columnes.
                            A festius s'indicara pais i regio o nomes pais.

                colinearity_remove_level - Per eliminar els atributs molt correlacionats entre ells. 0.9 eliminara atributs amb corr de pearson de mes de 0.9 deixant ne nomes 1. amb None no fara res.
                
                feature_selection - metode de seleccio d'atributs o reduccio utilitzat. 
                                    Pot ser:
                                        'Tree' -  utilitza un tree per descobrir els mes explicatius
                                        'PCA' - comprimeix X eliminant linearitats
                                         numero - Si es passa un numero enter es seleccionaran els KBest attributs 
                                         None - No fa res
                
                algorithm - l'algorisme que s'utilitzara per crear el model
                            Pot ser:
                                    'RF' = Random Forest
                                    'KNN' = KNN
                                    'SVR' = Super vector Regresor
                                    'MLP' = MLPRegressor
                                    'PLS' = PLSRegression
                
                params - Es un dict per pasar els parametres que es passaran a l'algorisme de forcasting utilitzat,
                         En el cas que sigui None i es fara un randomized search per buscar els parametres.

                escalat - Tipus d'escalat que s'aplica a les dades:
                                'MINMAX' = Minmax scaler
                                'Robust' = Robust scaler
                                'Standard' = Standard scaler
                
                max_time - Temps maxim en segons de comput per algorisme en mode cerca parametres. Altr. S'ignora.
            
            """
            
            # ja hauria d'estar fet pero per si de cas.
            #data = data.astype(float) --> no tenen perque ser tot floats!!! els trees accepten altres clases

            # Pas 1 - Fem el windowing
            dad = self.do_windowing(data, look_back)

            # Pas 2 - Creem variable dia_setmana, hora, mes
            dad =self.timestamp_to_attrs(dad, extra_vars)
            
            # Pas 3 - treiem colinearitats!
            [dad, to_drop] = self.colinearity_remove(dad, y, level=colinearity_remove_level)
            colinearity_remove_level_to_drop = to_drop

            # Pas 4 - treiem NaN! -- #TODO Fer opcional i permetre emplenar buits??
            dad.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = dad.dropna()

            # Pas 5 - desfem el dataset i ens quedem nomes amb les matrius X i y
            nomy = y
            y = X[nomy]
            del X[nomy]
            
            # Pas 6 - Escalat
            X, scaler = self.scalate_data(X, escalat)

            # Pas 7 - Seleccionem atributs
            [model_select, X_new, y_new] = self.treu_atrs(X, y, feature_selection)

            # Pas 8 - Creem el model
            [model, score] = self.Model(X_new, y_new.values, algorithm, params, max_time=max_time)

            ###
            #  Finalment un cop tenim el model configurat el guardem en un Object storer
            ###

            # Guardem els diferents models i configuracions que necessitarem per despres poder executar el forcast
            self.db['model'] = model
            self.db['scaler'] = scaler
            self.db['model_select'] = model_select
            self.db['colinearity_remove_level_to_drop'] = colinearity_remove_level_to_drop
            self.db['extra_vars'] = extra_vars
            self.db['look_back'] = look_back
            self.db['score'] = score
            self.db['objective'] = nomy
            
            if self.debug:  # m'interessa veure quan s'ha guardat un model, per saber per on va i que tot ha anat bé
                print('Model guardat!  Score:' + str(score))

        def forcast(self, data):
            """
            Funcio que fa la prediccio.
            
                data - dataframe amb timestamp a l'index, format de sempre. Ha de tenir tots els atributs inclosa la classe, ja que en el windowing necesitarem instancies passades de la classe i altres atributs, vigilar amb el windowing i l'historic que cal passar-li.
                
                les_id -id especifica del model, amb None s'utilitzaran noms de variables
            """
            
            ###
            # primer de tot - Recuperem el model

            model = self.db['model']
            model_select = self.db['model_select']
            scaler = self.db['scaler']
            colinearity_remove_level_to_drop = self.db['colinearity_remove_level_to_drop']
            extra_vars = self.db['extra_vars']
            look_back = self.db['look_back'] 
            y = self.db['objective']
            # variables = db.get_info('vars', les_id)

            """
            Ja tenim el model ara ens dediquem a transformar les dades perque quadrin amb el que s'ha fet a el model
            """
            
            # Pas 1 - Fem el windowing
            dad = self.do_windowing(data, look_back)        

            # Pas 2 - Creem variable dia_setmana, hora, mes?? -- #TODO Fer opcional!!
            dad = self.timestamp_to_attrs(dad, extra_vars)
            
            # Pas 3 - treiem colinearitats!
            if np.array(colinearity_remove_level_to_drop != None).any():
                dad.drop(colinearity_remove_level_to_drop, axis=1, inplace=True)
            
            # Pas 4 - eliminem la classe, ja hem posat els instants passats que necesitavem,
            # ens carreguem el que no necesitem
            del dad[y]
            
            ###
            # Pas 5 - treiem NaN! -- #TODO# Fer opcional i permetre emplenar buits??
            # Pero cuidado amb emplenar la classe, ja que aqui ens carreguem obs que no podem predir
            # perque no tenim instants passats!!!
            ###
            X = dad.dropna()

            # Pas 6 - Escalat
            if scaler is not None:
                x_i = pd.DataFrame(scaler.transform(X))
                X = x_i.set_index(X.index)
            
            # Pas 7 - Seleccionem atributs
            if model_select == []:
                X_new = X.values
            else:
                X_new = model_select.transform(X)
            # fem la prediccio i li posem l'index que toca en un dataframe
            out = pd.DataFrame(model.predict(X_new), columns=[y])
            out = out.set_index(X.index)
            
            # FI!!
            return out

        def timestamp_to_attrs(self, dad, extra_vars):

            ##
            # Creem variable dia_setmana, hora, mes
            ##

            if extra_vars is not None:
                # si es none no cal afegir res
                for i in extra_vars.keys():  # Per cada un dels keys del dict
                    if i == 'variables':  # si es afegir variables generades de l'index
                        for j in extra_vars[i]:
                            if j == 'Dia':
                                dad['Dia'] = dad.index.dayofweek
                            elif j == 'Hora':
                                dad['Hora'] = dad.index.hour
                            elif j == 'Mes':
                                dad['Mes'] = dad.index.month
                            elif j == 'Minut':
                                dad['Minut'] = dad.index.minute

                    elif i == 'festius':  # si es afegir variables generades de llibreria de festius
                        festius = extra_vars[i]
                        if len(festius) == 1:
                            import holidays
                            h = holidays.country_holidays(festius[0])
                            dad['festius'] = [x in h for x in dad.index.strftime('%Y-%m-%d').values]

                        if len(festius) == 2:
                            import holidays
                            h = holidays.country_holidays(festius[0], festius[1])
                            dad['festius'] = [x in h for x in dad.index.strftime('%Y-%m-%d').values]

            return dad

        def scalate_data(self, data, escalat):

            dad = data.copy()
            scaler = None
            if escalat != None:
                if escalat == 'MINMAX':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    scaler.fit(data)
                    dad = scaler.transform(data)
                elif escalat == 'Robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    scaler.fit(data)
                    dad = scaler.transform(data)

                elif escalat == 'Standard':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(data)
                    dad = scaler.transform(data)

                else:
                    raise ValueError('Undefined atribute selection method!')
            else:
                scaler = None

            return dad, scaler

        def save_model(self, filename='Model-data.joblib'):
            joblib.dump(self.db, filename)
            print("Model guardat al fitxer " + filename)

        def load_model(self, filename):
            self.db = joblib.load(filename)
