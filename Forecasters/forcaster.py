# Importem les llibreries necessàries
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import holidays
import warnings
warnings.filterwarnings('ignore')

class Forcaster:
        """
        Classe Forcaster per a la creació i gestió de models de predicció basats en dades temporals.

        Aquesta classe proporciona una sèrie de mètodes per:
        - Inicialitzar i configurar models de predicció.
        - Aplicar tècniques de "windowing" per crear variables addicionals basades en dades temporals.
        - Realitzar la selecció d'atributs per millorar el rendiment del model.
        - Entrenar models utilitzant diversos algorismes de machine learning.
        - Escalar dades per a la normalització o robustesa.
        - Realitzar prediccions sobre noves dades.
        - Desar i carregar models per a un ús posterior.

        Atributs:
            debug (bool): Indica si el mode de depuració està activat. Si és True, s'imprimiran missatges de depuració per ajudar en el desenvolupament.
            db (dict): Diccionari per emmagatzemar models creats, escaladors i configuracions necessàries.

        Mètodes:
            - __init__(debug): Inicialitza la classe amb opcions de depuració.
            - windowing_grup(datasetin, look_back_ini, look_back_fi): Crea variables de "windowing" per a un conjunt de dades.
            - windowing_univariant(datasetin, look_back_ini, look_back_fi, variable): Crea variables de "windowing" per a una variable especificada.
            - do_windowing(data, look_back): Aplica tècniques de "windowing" sobre el conjunt de dades.
            - colinearity_remove(datasetin, y, level): Elimina variables correlacionades segons un llindar especificat.
            - Model(X, y, algorithm, params, max_time): Realitza una cerca aleatòria per trobar la millor configuració per a un model.
            - treu_atrs(X, y, metode): Realitza la selecció d'atributs sobre el conjunt de dades.
            - create_model(data, y, look_back, extra_vars, colinearity_remove_level, feature_selection, algorithm, params, escalat, max_time): Crea i configura un model de predicció.
            - forcast(data): Realitza la predicció sobre les dades proporcionades.
            - timestamp_to_attrs(dad, extra_vars): Crea variables addicionals basades en el timestamp.
            - scalate_data(data, escalat): Escala les dades utilitzant diferents mètodes.
            - save_model(filename): Desa el model actual en un fitxer.
            - load_model(filename): Carrega un model des d'un fitxer.
        """

        def __init__(self, debug = False):
            """
            Constructor per defecte
            """

            self.debug = debug  # Opcions de depuració per mostrar per consola prints!
            self.search_space_conf_file = '../search_space.conf' # Fitxer de configuracio dels parametres a buscar en la cerca randomitzada.
            self.db = dict()  # Diccionari buit per emmagatzemar el model que hem creat.
        
        def windowing_grup(self, datasetin, look_back_ini=24, look_back_fi=48):
            """
            Funcio per crear les variables del windowing. Treballa sobre un dataset i inclou la variable objectiu!
            les variables creades es diran com l'original (legacy) i s'afegirà '_' i el numero de desplacament al final del nom.
            Es tindran en compte les hores en el rang [ini, fi)
            
            Parametres:
                datasetin - pandas dataframe amb datetime com a index
                
                look_back_ini - on comença la finestra numero d'observacions (24 -> el dia anterior si és horari)
                
                look_back_fi - fins on arriba numero d'observacions (48-< el dia anterior si és horari)
                

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
            '''
            Funció per aplicar el windowing a les variables d'un dataset, segons els valors especificats en el diccionari 'look_back'.
            Aquesta funció pot aplicar el finestratge a un grup de variables o individualment a cada variable. 
            Si es defineix una clau '-1' en el diccionari 'look_back', s'aplica a un grup de variables; la resta es fan individualment.
            Les variables tractades individualment es retornen amb les seves finestres desplaçades, afegint '_i' al final del nom, 
            on 'i' és el número de desplaçament.

            Paràmetres:
                data - pandas DataFrame amb les dades a processar.
                
                look_back - diccionari amb les finestres de temps. Cada clau és una variable (o '-1' per indicar grups), i el valor
                            és una tupla (ini, fi) que defineix el rang del windowing.

            Retorna:
                dad - un DataFrame amb les finestres aplicades a les variables especificades. Les variables individuals i les del grup
                    es gestionen segons les indicacions del diccionari.
            '''

            if look_back is not None:  # Retorna un array de NumPy, no una llista, des de l'objecte 'storer'.

            # Aplicar finestra (windowing) a les variables no especificades individualment
            if -1 in look_back.keys():  # Si l'indicador és -1, significa que volen fer un grup
                aux = look_back[-1]  # Recuperem els valors de la finestra per al grup

                # Recuperem les variables que es processaran individualment
                keys = list()
                for i in look_back.keys():
                    if i != -1:
                        keys.append(i)  # Afegim a la llista totes les variables excepte la del grup (-1)

                dad = data.copy()  # Copiem el dataset per preservar les variables individuals
                dad = dad.drop(columns=keys)  # Eliminem les variables que es faran individualment

                # Aplicar windowing al grup de variables
                dad = self.windowing_grup(dad, aux[0], aux[1])

                # Afegim les variables individuals que havíem tret
                for i in keys:
                    dad[i] = data[i]

            else:
                # Si no hi ha grups, totes les variables es processaran individualment
                dad = data.copy()  # Copiem el dataset
                keys = list()
                for i in look_back.keys():
                    if i != -1:
                        keys.append(i)  # Afegim les variables individuals

            # Aplicar windowing a les variables que es processen individualment
            variables = [col for col in data.columns if col not in keys]
            for i in variables:
                aux = look_back[-1]
                dad = self.windowing_univariant(dad, aux[0], aux[1], i)

        else:
            # Si no hi ha windowing, simplement es copia el dataset original
            dad = data.copy()

        return dad


        def colinearity_remove(self, datasetin, y, level=0.9):
            """
            Elimina les variables del DataFrame que estan massa correlacionades (colinearitat) segons un llindar de correlació de Pearson.
            
            Paràmetres:
                datasetin - pandas DataFrame amb datetime com a index.
                
                y - la variable objectiu (per assegurar-nos que no l'eliminem durant el procés).
                
                level - llindar de correlació de Pearson per eliminar variables (per defecte 0.9). Si és None, no es fa cap eliminació.
                
            Retorna:
                dataset - El dataset amb les variables correlacionades eliminades.
                to_drop - Llista de les variables que han estat eliminades.
            """

            if level is None:
                dataset = datasetin  # Si no hi ha llindar, retornem el dataset original sense modificar res.
                to_drop = None  # No eliminem cap variable.
            else:
                # Calcular la matriu de correlació i seleccionar les variables massa correlacionades.
                corr_matrix = datasetin.corr().abs()  # Matriu de correlació amb valors absoluts.
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Agafem només la part superior de la matriu.
                
                # Crear una llista de les columnes a eliminar (aquelles amb correlació superior al nivell donat).
                to_drop = [column for column in upper.columns if any(upper[column] > level)]
                
                # Ens assegurem que no eliminem la variable objectiu (y) si està a la llista 'to_drop'.
                if np.array(to_drop == y).any():
                    del to_drop[to_drop == y]  # Eliminem 'y' de la llista 'to_drop', si hi és.
                
                # Eliminem les columnes correlacionades del dataset.
                datasetin.drop(to_drop, axis=1, inplace=True)
                dataset = datasetin.copy()  # Fem una còpia del dataset resultant.
            
            return [dataset, to_drop]  # Retornem el dataset modificat i les variables eliminades.

        
        def Model(self, X, y, algorithm='RF', params=None, max_time=None):
            """
            Funció que realitza una cerca aleatòria (randomized search) per trobar una bona configuració d'un algoritme de predicció
            indicat, o directament crea el model amb els paràmetres especificats.

            Paràmetres:
                X - np.array amb les dades d'entrenament.
                y - np.array amb la variable objectiu (target).
                algorithm - String o llista d'algorismes a provar. Per defecte 'RF' (Random Forest).
                params - Diccionari de paràmetres específics per a l'algorisme. Si és None, es fa una cerca aleatòria.
                max_time - Temps màxim per a la cerca (en segons). Si és None, es fan totes les iteracions possibles.

            Retorna:
                model - El millor model trobat després de la cerca aleatòria o amb els paràmetres especificats.
                score - El millor valor del Mean Absolute Error (MAE) obtingut durant la cerca.
            """
            
            ## Primer carreguem el grid de paràmetres des del fitxer i fem els imports necessaris
            import json
            with open(self.search_space_conf_file) as json_file:
                d = json.load(json_file)
            
            if params is None:
                # Si no tenim paràmetres específics, fem la cerca aleatòria

                # Preparem les dades de train i test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # No fem shuffle per mantenir la seqüència temporal.
                
                # Inicialitzem llibreries i el millor MAE (inicialment infinit)
                from sklearn.model_selection import ParameterSampler
                from sklearn.metrics import mean_absolute_error
                import time
                best_mae = np.inf  # Inicialitzem el millor MAE com a infinit per trobar-ne un millor.
                
                # Preparem la llista d'algorismes a provar (tots o el que ens indiquen)
                if algorithm is None:
                    # No s'ha especificat un algoritme, els provem tots
                    algorithm_list = list(d.keys())
                elif isinstance(algorithm, list):
                    # S'ha passat una llista d'algorismes a provar
                    algorithm_list = algorithm
                else:
                    # S'ha passat un únic algoritme
                    algorithm_list = [algorithm]
                
                # Per cada algoritme a provar
                for i in range(len(algorithm_list)):
                    # Recuperem el grid de paràmetres per l'algoritme actual
                    random_grid = d[algorithm_list[i]][0]
                    
                    if max_time is None:
                        iters = d[algorithm_list[i]][1]  # Iteracions per defecte
                    else:
                        iters = max_time  # Iteracions limitades per temps
                    
                    # Strings per fer l'import corresponent a l'algoritme
                    impo1 = d[algorithm_list[i]][2]
                    impo2 = d[algorithm_list[i]][3]

                    if self.debug:
                        print(" ")
                        print("Començant a optimitzar: " + algorithm_list[i] + 
                            ' - Algorisme ' + str(algorithm_list.index(algorithm_list[i])+1) + 
                            ' de ' + str(len(algorithm_list)) + 
                            ' - Temps màxim aproximat (segons): ' + str(iters))

                    # Preparem una mostra aleatòria de paràmetres
                    sampler = ParameterSampler(random_grid, n_iter=np.iinfo(np.int64).max)
                    
                    # Importem la llibreria correcta per a l'algoritme actual
                    a = __import__(impo1, globals(), locals(), [impo2])
                    Forcast_algorithm = eval("a."+impo2)
                    
                    try:
                        # Creem i avaluem els models un a un
                        t = time.perf_counter()
                        if self.debug:
                            print("Màxim de " + str(len(sampler)) + " configuracions a provar!")
                            j = 0
                        
                        for params in sampler:
                            regr = Forcast_algorithm(**params)
                            pred_test = regr.fit(X_train, y_train).predict(X_test)
                            act = mean_absolute_error(y_test, pred_test)
                            if best_mae > act:
                                best_model = regr
                                best_mae = act

                            if self.debug:
                                print("\r", end="")
                                j += 1
                                print(str(j) + "/" + str(len(sampler)) + 
                                    " Millor MAE: " + str(best_mae) + 
                                    " (" + type(best_model).__name__ + 
                                    ") Últim MAE: " + str(act) + 
                                    " Temps transcorregut: " + str(time.perf_counter() - t) + " s", end="")
                                
                            if (time.perf_counter() - t) > iters:
                                if self.debug:
                                    print("Algoritme " + algorithm_list[i] + ' - Temps màxim de cerca assolit.')
                                break
                    except Exception as e:
                        print("Warning: Algoritme " + algorithm_list[i] + ' - Abortat per error: ' + str(e))
                
                # Finalment ajustem el millor model trobat amb tot el conjunt de dades
                best_model.fit(X, y)
                model = best_model
                score = best_mae
                
                return [model, score]
    
            else:
                # Si s'especifica un algoritme i paràmetres concrets
                try:
                    # Recuperem els strings per fer l'import corresponent a l'algoritme
                    impo1 = d[algorithm][2]
                    impo2 = d[algorithm][3]
                except:
                    raise ValueError('Algoritme de predicció no definit!')
                
                # Importem la llibreria correcta
                a = __import__(impo1, globals(), locals(), [impo2])
                Forcast_algorithm = eval("a." + impo2)
                
                # Ajustem el model amb els paràmetres indicats
                f = Forcast_algorithm()
                f.set_params(**params)
                f.fit(X, y)
                score = 'none'  # No es calcula cap mètrica en aquest cas
                return [f, score]


        # Feature selection/reduction methods
        def treu_atrs(self, X, y, metode=None):
            """
            Fem una selecció o reducció d'atributs de les dades d'entrada.

            Paràmetres:
                X - np.array amb les dades d'entrenament (features)
                y - np.array amb la variable objectiu (target)
                
                metode - Mètode per seleccionar o reduir atributs:
                        None    -> No es fa cap selecció, es retornen els atributs originals.
                        'Tree'  -> Utilitza un ExtraTreesRegressor per seleccionar atributs rellevants.
                        integer -> Selecciona el nombre de features indicat mitjançant un test estadístic.
                        'PCA'   -> Aplica PCA (Anàlisi de Components Principals) per reduir la dimensionalitat.

            Retorna:
                model_select - L'objecte del model de selecció o reducció d'atributs utilitzat.
                X_new        - El conjunt de dades amb els atributs seleccionats o reduïts.
                y            - La variable objectiu (es retorna sense canvis).
            """

           # Si no es especifica cap mètode, retornem les dades originals
            if metode is None:
                model_select = []  # No hi ha model de selecció
                X_new = X  # Retornem les dades originals
            elif metode == 'Tree':
                # Utilitzem un model d'arbre extra per seleccionar atributs
                from sklearn.ensemble import ExtraTreesRegressor
                from sklearn.feature_selection import SelectFromModel
                
                clf = ExtraTreesRegressor(n_estimators=50)  # Inicialitzem el model amb 50 estimadors
                clf = clf.fit(X, y)  # Ajustem el model a les dades
                model_select = SelectFromModel(clf, prefit=True)  # Creem l'objecte de selecció d'atributs
                X_new = model_select.transform(X)  # Transformem X per obtenir només els atributs seleccionats
            elif type(metode) is int:                
                # Seleccionem un nombre fixe d'atributs
                from sklearn.feature_selection import SelectKBest, f_classif
                
                model_select = SelectKBest(f_classif, k=metode)  # Inicialitzem SelectKBest amb el mètode f_classif
                X_new = model_select.fit_transform(X, y)  # Ajustem i transformem X
            elif metode == 'PCA':
                # Aplica PCA per reduir la dimensionalitat
                from sklearn.decomposition import PCA
                
                model_select = PCA(n_components='mle')  # Estimem la dimensió òptima
                X_new = model_select.fit_transform(X)  # Transformem X
            else:
                # Si el mètode no és vàlid, llançem una excepció
                raise ValueError('Mètode de selecció d\'atributs no definit!')
                        
            # Retornem l'objecte de selecció, les noves dades i la variable objectiu
            return [model_select, X_new, y]


        #A partir d'aqui tenim les 2 funcions que controlen tot el funcionament del forcasting 
        # create_model() - crear i guardar el model
        # forcasting() - recuperar i utilitzar el model 
        def create_model(self, data, y, look_back={-1:[25,48]}, extra_vars={'variables':['Dia','Hora','Mes'], 'festius':['ES','CT']},
                         colinearity_remove_level=0.9, feature_selection='Tree', algorithm='RF', params=None, escalat=None, max_time=None):
            """
            Funció per crear, guardar i configurar el model de forecasting.

            Paràmetres:
                data - pandas DataFrame amb datetime com a índex, format de sempre...
                
                y - nom de la columna amb la variable objectiu
                
                look_back - Diccionari per aplicar windowing:
                    - Clau: nom de la columna a aplicar el windowing.
                    - Valor: finestra a aplicar [ini, fi] o [ini, fi, salt].
                    Exemple: 
                    {'y':[25,48], -1:[1,480,24]} # 'y' de 25 a 48 observacions, totes les altres de 1 a 480 amb salt de 24.
                
                extra_vars - Diccionari per crear variables noves. Claus: 'variables' per les generades des de l'índex o 'festius'.
                    Exemple: {'variables':['dia','hora','mes'],'festius':['ES','CT']}
                
                colinearity_remove_level - Nivell per eliminar atributs molt correlacionats (0.9 elimina atributs amb corr de Pearson > 0.9).
                
                feature_selection - Mètode de selecció o reducció d'atributs:
                    - 'Tree': utilitza un arbre per descobrir atributs més explicatius.
                    - 'PCA': comprimeix X eliminant linearitats.
                    - número: selecciona els KBest atributs.
                    - None: no fa res.
                
                algorithm - Algorisme a utilitzar per crear el model:
                    - 'RF': Random Forest
                    - 'KNN': KNN
                    - 'SVR': Super Vector Regressor
                    - 'MLP': MLPRegressor
                    - 'PLS': PLSRegression
                
                params - Diccionari per passar paràmetres a l'algorisme de forecasting. Si és None, es fa un randomized search per buscar paràmetres.
                
                escalat - Tipus d'escalat a aplicar a les dades:
                    - 'MINMAX': Minmax scaler
                    - 'Robust': Robust scaler
                    - 'Standard': Standard scaler
                
                max_time - Temps màxim en segons de comput per l'algorisme en mode cerca de paràmetres. Altres temps s'ignoren.
            """
            
            # Pas 1 - Fem el windowing
            dad = self.do_windowing(data, look_back)

            # Pas 2 - Creem variables dia_setmana, hora, mes
            dad = self.timestamp_to_attrs(dad, extra_vars)
            
            # Pas 3 - Eliminem colinearitats
            [dad, to_drop] = self.colinearity_remove(dad, y, level=colinearity_remove_level)
            colinearity_remove_level_to_drop = to_drop

            # Pas 4 - Eliminem NaN
            dad.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = dad.dropna()

            # Pas 5 - Desfem el dataset i ens quedem només amb les matrius X i y
            nomy = y
            y = X[nomy]
            del X[nomy]
            
            # Pas 6 - Escalat
            X, scaler = self.scalate_data(X, escalat)

            # Pas 7 - Seleccionem atributs
            [model_select, X_new, y_new] = self.treu_atrs(X, y, feature_selection)

            # Pas 8 - Creem el model
            [model, score] = self.Model(X_new, y_new.values, algorithm, params, max_time=max_time)

            # Guardem els diferents models i configuracions necessàries per executar el forecasting posteriorment
            self.db['model'] = model
            self.db['scaler'] = scaler
            self.db['model_select'] = model_select
            self.db['colinearity_remove_level_to_drop'] = colinearity_remove_level_to_drop
            self.db['extra_vars'] = extra_vars
            self.db['look_back'] = look_back
            self.db['score'] = score
            self.db['objective'] = nomy
            
            if self.debug:  # M'interessa saber quan s'ha guardat un model per seguir el progrés
                print('Model guardat!  Score:' + str(score))

        def forcast(self, data):
            """
            Funció que realitza la predicció sobre les dades proporcionades.
            
            Paràmetres:
                data - DataFrame amb timestamps a l'índex, que ha de tenir el format habitual.
                    Ha d'incloure tots els atributs necessaris, incloent la classe,
                    ja que en el procés de windowing es necessitaran instàncies passades de la classe
                    i altres atributs. Cal vigilar amb el windowing i l'històric que es necessita passar.
            """
            
            # Pas 0 - Recuperem el model i les configuracions necessàries de la base de dades
            model = self.db['model']
            model_select = self.db['model_select']
            scaler = self.db['scaler']
            colinearity_remove_level_to_drop = self.db['colinearity_remove_level_to_drop']
            extra_vars = self.db['extra_vars']
            look_back = self.db['look_back'] 
            y = self.db['objective']

            # Ara que tenim el model, transformarem les dades per assegurar-nos que coincideixin amb les que s'han utilitzat durant la creació del model.

            # Pas 1 - Aplicar windowing a les dades
            dad = self.do_windowing(data, look_back)

            # Pas 2 - Crear variables addicionals (dia de la setmana, hora, mes)
            # TODO: Fer opcional
            dad = self.timestamp_to_attrs(dad, extra_vars)
            
            # Pas 3 - Eliminar atributs colineals
            if colinearity_remove_level_to_drop is not None:
                dad.drop(colinearity_remove_level_to_drop, axis=1, inplace=True)
            
            # Pas 4 - Eliminar la classe del DataFrame, ja que hem afegit els instants passats necessaris
            del dad[y]
            
            ### 
            # Pas 5 - Eliminar NaN
            # TODO: Fer opcional i permetre emplenar buits
            # Important: no emplenar la classe, ja que això podria eliminar observacions que no podem predir
            X = dad.dropna()

            # Pas 6 - Escalar les dades si el scaler és proporcionat
            if scaler is not None:
                x_i = pd.DataFrame(scaler.transform(X))
                X = x_i.set_index(X.index)
            
            # Pas 7 - Seleccionar atributs
            if model_select == []:
                X_new = X.values
            else:
                X_new = model_select.transform(X)

            # Realitzar la predicció i col·locar l'índex adequat en un nou DataFrame
            out = pd.DataFrame(model.predict(X_new), columns=[y])
            out = out.set_index(X.index)
            
            # Fi de la funció: retornem el DataFrame amb les prediccions
            return out

        def timestamp_to_attrs(self, dad, extra_vars):
            """
            Funció que afegeix atributs addicionals al DataFrame basant-se en els timestamps de l'índex.
            
            Paràmetres:
                dad - DataFrame que conté les dades amb un índex de timestamps.
                extra_vars - Diccionari que especifica quines variables addicionals es volen crear.
                            Pot contenir dues claus:
                            - 'variables': una llista de variables a generar a partir de l'índex.
                            - 'festius': una llista de països (i opcionalment regions) per identificar els dies festius.

            Retorna:
                dad - DataFrame amb les noves variables afegides.
            """
            
            # Creem variables addicionals (dia de la setmana, hora, mes, minut) basades en l'índex de timestamps.

            if extra_vars is not None:
                # Si no hi ha extra_vars, no s'afegeix cap variable nova.
                for key in extra_vars.keys():  # Iterem per cada clau del diccionari extra_vars.
                    if key == 'variables':  # Si es volen afegir variables basades en l'índex.
                        for var in extra_vars[key]:
                            # Afegim les variables corresponents basant-nos en l'índex de timestamps.
                            if var == 'Dia':
                                dad['Dia'] = dad.index.dayofweek  # Dia de la setmana (0 = dilluns, 6 = diumenge).
                            elif var == 'Hora':
                                dad['Hora'] = dad.index.hour  # Hora del dia.
                            elif var == 'Mes':
                                dad['Mes'] = dad.index.month  # Mes de l'any.
                            elif var == 'Minut':
                                dad['Minut'] = dad.index.minute  # Minut del dia.

                    elif key == 'festius':  # Si es volen afegir variables de festius.
                        festius = extra_vars[key]
                        import holidays
                        
                        # Comprovem quants països es proporcionen.
                        if len(festius) == 1:
                            # Afegim festius per un sol país.
                            h = holidays.country_holidays(festius[0])
                            dad['festius'] = [date in h for date in dad.index.strftime('%Y-%m-%d').values]

                        elif len(festius) == 2:
                            # Afegim festius per un país i una regió.
                            h = holidays.country_holidays(festius[0], festius[1])
                            dad['festius'] = [date in h for date in dad.index.strftime('%Y-%m-%d').values]

            return dad  # Retornem el DataFrame actualitzat amb les noves variables.

        def scalate_data(self, data, escalat):
            """
            Funció que aplica un escalat a les dades proporcionades.

            Paràmetres:
                data - DataFrame o array amb les dades a escalar.
                escalat - Tipus d'escalat a aplicar. Pot ser:
                        - 'MINMAX': Escalat entre 0 i 1.
                        - 'Robust': Escalat robust que redueix la influència dels valors atípics.
                        - 'Standard': Escalat a mitjana 0 i desviació estàndard 1.

            Retorna:
                - dad: Dades escalades.
                - scaler: L'objecte scaler utilitzat per aplicar l'escalat, o None si no s'ha aplicat escalat.
            """

            dad = data.copy()  # Fem una còpia de les dades originals per no modificar-les.
            scaler = None  # Inicialitzem l'escala com a None.

            if escalat is not None:  # Comprovem si s'ha especificat un tipus d'escalat.
                if escalat == 'MINMAX':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()  # Creem una instància de MinMaxScaler.
                    scaler.fit(data)  # Ajustem l'escaler a les dades.
                    dad = scaler.transform(data)  # Transformem les dades.
                
                elif escalat == 'Robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()  # Creem una instància de RobustScaler.
                    scaler.fit(data)  # Ajustem l'escaler a les dades.
                    dad = scaler.transform(data)  # Transformem les dades.

                elif escalat == 'Standard':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()  # Creem una instància de StandardScaler.
                    scaler.fit(data)  # Ajustem l'escaler a les dades.
                    dad = scaler.transform(data)  # Transformem les dades.

                else:
                    # Si s'especifica un tipus d'escalat desconegut, llencem un error.
                    raise ValueError('Undefined attribute selection method!')
            else:
                scaler = None  # Si no s'ha especificat escalat, mantenim scaler com a None.

            return dad, scaler  # Retornem les dades escalades i l'objecte scaler.

        def save_model(self, filename='Model-data.joblib'):
            """
            Funció que guarda el model en un fitxer utilitzant joblib.

            Paràmetres:
                filename - Nom del fitxer on es guardarà el model. Per defecte és 'Model-data.joblib'.

            Retorna:
                No retorna res, però imprimeix un missatge confirmant que el model ha estat guardat.
            """
            
            joblib.dump(self.db, filename)  # Guardem la base de dades del model en el fitxer especificat.
            print("Model guardat al fitxer " + filename)  # Imprimim un missatge de confirmació.

        def load_model(self, filename):
            """
            Funció que carrega un model des d'un fitxer utilitzant joblib.

            Paràmetres:
                filename - Nom del fitxer d'on es carregarà el model.

            Retorna:
                No retorna res, però actualitza l'atribut db amb les dades del model carregat.
            """

            self.db = joblib.load(filename)  # Carreguem el model des del fitxer especificat i actualitzem la base de dades.
