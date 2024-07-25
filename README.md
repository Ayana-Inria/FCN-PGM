# FCN-PGM


Classificazione di dati multisensore alla stessa data (WP 7101, SW04)

----------------------------
---- CONTENUTO Zip-FILE ----
----------------------------

WP7101.zip/
	datasets/
		_HOW_TO_WRITE_ONE.json
		datasets.py
		Milan_SAR.json
	experiments/
		_HOW_TO_WRITE_ONE.json
		unet_mpm.json
	input/
		gt/
			DUSAF6.shp
		final_dataset/ (esempio di dataset usato)
			3channels/
				img/
				   (immagini Cosmo-SkyMed, SAOCOM, Cosmo-SkyMed + SAOCOM registrate con il WP 1103 e divise in sotto-immagini per l'addestramento della rete)
				gt/
				   (gt raster alla risoluzione delle immagini Cosmo-SkyMed di cui sopra)
			README.txt
	model/
		mpmOnQuadtree/
			__init__.py
			hilbertCurve.py
			method.py
		ensemble_estimation.py
		extract_activations.py
	net/
		loss.py
		train.py
		unet.py
	output/
		model_final
		model_mulires_final
	utils/
		export_result.py
		utils.py
		utils_dataset.py
		utils_network.py
		utils_quadtree.py
	create_dataset.py
	multimission_classification.py
	README.txt
	requirements.txt
	package-list.txt



Cartella "WP7101" contenente i moduli Python relativi al SW04.
Cartella "datasets" contiene i file json per descrivere il dataset e il file .py per creare un dataloader per addestrare la rete neurale.  
Cartella "experiments" contiene i file json per predisporre l'esperimento.
Cartella "input" contiene un esempio di dataset multisensore, si fa riferimento al file input/final_dataset/README.txt
Cartella "model" contiene il modello grafico probabilistico (il PGM gerarchico).
Cartella "net" contiene la rete neurale.
Cartella "output" contiene due esempi di addestramento di rete neurale (singola risoluzione e multirisoluzione) e conterra' i risultati degli esperimenti.
Cartella "utils" contiene le funzioni necessarie al funzionamento dei moduli contenuti nelle altre cartelle.

------------------------------------
---- REQUISITI DI INSTALLAZIONE ----
------------------------------------

Python 3.8
Anaconda

La lista dei pacchetti utilizzati nel'ambiente virtuale e' contenuta nel file "environment.yml". Puo' essere utilizzata per creare un nuovo ambiente virtuale tramite:

 conda env create -f environment.yml

------------------
---- UTILIZZO ----
------------------


\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


usage: multimission_classification.py [-h] [-i INPUT_DIR_PATH]
                                      [-o OUTPUT_DIR_PATH] [-r] [-mul]
                                      [-w [WINDOW_SIZE]] [-b [BATCH_SIZE]]
                                      [-exp [EXPERIMENT_NAME]] [-lr BASE_LR]
                                      [-d [DATASET]] [-e EPOCHS]
                                      [-se [SAVE_EPOCH]]

Multimission image classification

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR_PATH, --input INPUT_DIR_PATH
                        Path of input directory
  -o OUTPUT_DIR_PATH, --output OUTPUT_DIR_PATH
                        Path of output directory
  -r, --retrain         Retrain the neural network
  -mul, --multires      Train the network with multiresolution input data
  -w [WINDOW_SIZE], --window [WINDOW_SIZE]
                        Dimension of the crops of the images input to the
                        network. Default is (256, 256)
  -b [BATCH_SIZE], --batch_size [BATCH_SIZE]
                        Size of the image batch input to the network. Default
                        is 10
  -exp [EXPERIMENT_NAME], --experiment_name [EXPERIMENT_NAME]
                        Experiment_name identify which experiment to run. Must
                        match the .json file in the experiments folder.
  -lr BASE_LR, --base_lr BASE_LR
                        Base learning rate of the neural network
  -d [DATASET], --dataset [DATASET]
                        Dataset name. If set to Milano, use the dataset configuration
                        in datasets/
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs of the training of the neural network
  -se [SAVE_EPOCH], --save_epoch [SAVE_EPOCH]
                        When to save the model


Ulteriori commenti su alcuni dei parametri:

i: percorso della directory in input (parametro facoltativo, il valore di default e' la cartella con l'esempio del dataset "./input/final_dataset/3channels/")
o: percorso della directory in output (parametro facoltativo, il valore di default e' "./output/")
r: flag per riaddestrare la rete da capo (parametro facoltativo, default e' False). Se False occorre caricare un modello gia' addestrao (contenuto in "./output/")
mul: flag per utilizzare il modello inserendo immagini multisensore a diversa risoluzione (parametro facoltativo, default e' False). Se True occorre che le immagini SAOCOM 
     abbiano una risoluzione pari alla meta' di quelle Cosmo-SkyMed

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Esempio riga di comando per il lancio del modulo (Windows OS): 

python multimission_classification.py -i "./input/final_dataset/3channels/" -o "./output/" -r 
python multimission_classification.py -i "./input/final_dataset/3channels/" -o "./output/" -r -mul


\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

-----------------
---- DATASET ----
-----------------

La cartella WP7101/input/ coontiene gia' un esempio di dataset utilizzato per la validazione sperimentale. In caso si volesse effettuare una nuova validazione sperimentale
a partire da altri dati pre-processati con il SW01, occorre utilizzare modulo seguente "create_dataset.py", per ottenere training e test set per l'addestramento del metodo 
di classificazione multisensore (FCN + PGM).

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


usage: create_dataset.py [-h] -i INPUT_IMG_PATH [-o OUTPUT_DIR_PATH] [-r]
                         [-p RASTER_GT_PATH] [-w [WINDOW_SIZE]] -t
                         {sao,csk,gt,stack}



Create dataset for multimission image classification

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_IMG_PATH, --input_raster_img INPUT_IMG_PATH
                        Path of input image to use as reference for the
                        rasterization of the GT and to crop.
  -o OUTPUT_DIR_PATH, --output OUTPUT_DIR_PATH
                        Path where the new dataset will be stored. Default
			is "./input/new_dataset/"
  -r, --gt_to_rasterize
                        Rasterize the GT wrt the raster input image (ex. a CSK
                        image after coregistration)
  -p RASTER_GT_PATH, --raster_gt_path RASTER_GT_PATH
                        Path where to save the rasterized GT
  -w [WINDOW_SIZE], --window_size [WINDOW_SIZE]
                        Dimension of the cropped images. Default is (1024,
                        1024)
  -t {sao,csk,gt,stack}, --type {sao,csk,gt,stack}
                        Type of the image to crop.


Ulteriori commenti su alcuni dei parametri:

i: Percorso dell'immagine da usare per creare il nuovo set per gli esperimenti (e da usare per creare un raster della realta' al suolo DUSAF6.shp, se aggiunto il flag -r).
   L'immagine puo' essere (pre-processing tramite il SW01: clipping, resampling, registering):
	- SAOCOM (coregistrata con un'immagine Cosmo-SkyMed, scelta affinche' le date di acquisizione siano il piu' vicino possibili, ricampionata ad una risoluzione pari alla meta' della risoluzione dell'immagine Cosmo)
	- Cosmo-SkyMed (coregistrata con un'immagine SAOCOM come sopra, ricampionata ad una risoluzione pari al doppio della risoluzione dell'immagine SAOCOM)
	- una concatenazione delle due precedenti, coregistrate e ricampionate alla stessa risoluzione
   Si suggerisce di creare il raster della GT alla risoluzione piu' fine (quindi usando l'immagine Cosmo-SkyMed in concomitanza al flag -r). 
r: flag per rasterizzare la GT contenuta in "./input/gt/DUSAF6.shp" (parametro facoltativo, di default e' False)
p: Percorso in cui salvare la GT raterizzata (parametro facoltativo, di default e' "./input/gt/rasterized_gt.tif")
w: Dimensione finale delle immagini che compongono training e test set della rete neurale (parametro facoltativo, default e' (1024, 1024)). Se si vuole usare il modello in modalita' MULTIRISOLUZIONE
   (con immagini Cosmo-SkyMed a risoluzione X e immagini SAOCOM a risoluzione X/2) occorre che la 'w' delle immagini a risoluzione X/2 sia la meta' di quella delle immagini a risoluzione X
t: Definisce il nome delle immagini del training e test set. Se 'stack' occorre concatenare immagini Cosmo-SkyMed e SAOCOM alla stessa risoluzione


\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


Esempio riga di comando per il lancio del modulo (Windows OS): 

python create_dataset.py -i "/PATH/TO/COREGISTERED_IMAGE_WITH_SW01" -t csk
