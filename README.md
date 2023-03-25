# A feladat megoldása:
A feladat megoldásához egy egyszerű CLI-t hoztam létre, mely segítségével elsősorban állíthatóak a model paraméterei, a bemenő változók a modellhez, a train-test split aránya a dataset elérési útvonala és a kimeneti fájlok mentési útvonala.

A megoldás során a model a train adathalmaz alapján tanul és a test halmaz alapján ad predikciókat számunkra. A teszt halmazt és a predikciókat használtam ahhoz, hogy különböző output file-okat generáljak.

## Kimenetett fájlok:
- evaluation_metrics: A regressziós feladathoz legtöbbször alkalmazott RMSE, MSE, MAE és R^2 metrikákat a program kiszámolja és ezeket egy egyszerű .csv formátumú fájlba menti el.

- evaluation_values: Ebben fájlban találhatóak a valós teszt értékek és a kimenetként kapott eredmények egymás mellett. 

- --round True: Ha a CLI-ban  a --round opciót True-ra állítjuk akkor mivel a feladat nagyon hasonlít egy klasszifikációs feladathoz kerekít a közelített értékeken, hogy könnyebbé váljon a model kirtékelése. Ekkor egy pontosságot (accuracy) is számít az evluation_metrics file-ban illetve a kerekített értékeket meg is jeleníti az evaluation_values file-ban.

## Ábrák: 
Minden esetben kapunk ábrákat, melyeket elsősorban matplotlibbel készítettem el. Az alap ábrák mikor nem adunk meg más specifikációt és 2-nél több változót használunk a következők:

- Residual plot: A regresszió után kivonjuk a predikált értéket a valós értékekből majd megjelenítjük őket az y tengelyen, az x tengelyen pedig a valós értékeket (a célváltozót). A pontok által leírt alakból következtethetünk arra, hogy mennyire jó a regressziós modellünk. 

- Megjelenítjük a predikált értékeket a valós értékek függvényében. Ha a pontok az y=x egyenes körül vannak szorosan akkor jó a regressziós modellünk. 

- Residual histogram: A maradékok eloszlását is megfigyelhetjük. A histogram alakjának normál eloszlásúnak kell lennie.

### További ábrák: 
Ha további specifikációkat adunk meg akkor több ábrát is kaphatunk amivel jobban bele láthatunk a model működésébe. 

- --round True: Konfúziós Mátrix heatmap-en ábrázolva - Ha kerekítjük a predikált értékeket akkor a program kirajzol számunkra egy konfúziós mátrixot melyen látható, melyek azok az értékek amelyet a model nem tudott megfelelően eltalálni. 
- Bemenő változók száma = 1. Ha csak 1 bemenő változónk van akkor a program készít egy ábrát, amin az x tengelyen a bemenő változó az y tengelyen a célváltozó lesz látható. Továbbá a modell által predikált értékeket is mutatja egy másik színnel. Itt szintén látszik majd, mely eseteket nem sikerült megfelelően kezelni a modellnek.
- Bemenő Változók száma = 2. Ebben az esetben egy 3D koordináta rendszerben ábrázolt ábrát kapunk ahol az x y tengelyen a bemenő és a z tengelyen a célváltozót láthatjuk majd. Illetve azt a síkot amelyet a model a két változóból predikált.

## HTML report:
Amennyiben megadjuk a --html_report True értéket a program argumentumába úgy a program generál egy HTML fájlt, amelyben interaktívan érhetőek el a fent említett ábrák illetve a kimentett táblázatokat is megfigyelhetjük. Így kapunk egy könnyen elérhető és interaktív felületet a model kiértékelésére.

## Docker:
A feladat említett bónusz pontokat amennyiben Docker konténerizált a feladat. A root mappában található egy Dockerfile illetve egy docker-compose.yml is. 
A Docker image-ben csak a main.py file van benne, amiben a feladat során elkészítettem a megoldásom. A docker container inicializálásakor kapni fogunk egy CLI-t ahonnan a main.py futtatható mintha rendes terminálon keresztül futtatnánk. A .yml file főleg azért készült el hogy el tudjuk érni a dataset-et a host gépről, illetve hozzáadtam egy output_docker nevű mappát is ahová (opcionálisan) menthetjük majd ki az output-okat (ábrák, .csv file-ok és a HTML report).
Ha szükséges feltölthetem Docker Hub-ra a felépített Docker Image-t azonban a build is könnyen futtatható. 

Instrukciók:

építsük a docker image-t

```
docker build -t python_cli --rm .
```

Inicializáljuk a Docker Container-t docker-compose-al

```
docker-compose run --rm app
```
Így futtathatjuk is az applikációt pl. a következőképp:
```
python main.py -fp dataset/winequality.csv -op output/ -r True -hr True
```

##A program argumentumai:
optional arguments:
  -h, --help            show this help message and exit
  -fp FILEPATH, --filepath FILEPATH
                        Type in the filepath of the .csv file with your dataset. Default is :'winequality.csv'
  -l LABEL, --label LABEL
                        name of the column that contrains the label (the y, or the value we are predicting). Default is : quality
  -tf [TRAIN_FEATURES ...], --train_features [TRAIN_FEATURES ...]
                        Features (columns of csv) for the model to make the predictions. Default uses all. Write your columns names afer each other. e.g. -mf   
                        pH sulphates. If there the feature consists of multiple words write it in double quotes like this: "fixed acidity". If the number of    
                        feature is one or two it will create visualizations for those (2D if only one feature is present, 3D if two features are present)       
  -f {True,False}, --features {True,False}
                        print out the features. Default is False
  -s SPLIT, --split SPLIT
                        Thepercentage of the train-test split (the value you pass will give you the train percentage). Default is: 0.25
  -op OUTPUT_PATH, --output_path OUTPUT_PATH
                        Specify the path of the output file. It will contain the evaluation metrics, and the predicted values with true values in a separate    
                        file. Default is the currect dir
  -r ROUND, --round ROUND
                        Round the results (True or False). If True will automatically create a confusion matrix plot and calculate the accuracy of the model.   
                        Default is
  -sk {linear,poly,rbf,sigmoid,precomputed}, --svm_kernel {linear,poly,rbf,sigmoid,precomputed}
                        The typeof the kernel for the SVR model. Default is rbf
  -de DEGREE, --degree DEGREE
                        The degree of the polynomial if the kernel is 'poly' (ignored by other kernels). Default is 3.
  -ga {scale,auto}, --gamma {scale,auto}
                        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
  -co COEF0, --coef0 COEF0
                        Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'. Default is 0.0
  -to TOL, --tol TOL    Tolerance for stopping criterion. Default is 1e-3
  -c C, --C C           Regularization parameter The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is   
                        a squared l2 penalty. Default is 1.0
  -e EPSILON, --epsilon EPSILON
                        - SVR model epszilon értéke. Nem negatív szám. Default = 0.1.
  -sh SHRINKING, --shrinking SHRINKING
                        - csökkenő heurisztika alkalmazása. Default = True
  -cs CACHE_SIZE, --cache_size CACHE_SIZE
                        - A kernel cache mérete mb-ban. Default = 200.0
  -v VERBOSE, --verbose VERBOSE
                       - Az output megjelenítsére szolgáló paraméter. Default = False
  -mi MAX_ITER, --max_iter MAX_ITER
  - Az SVR-re adott hard limit faktor. Default = -1
  -hr , --html_report 
  - A HTML file mentése. Default = False
