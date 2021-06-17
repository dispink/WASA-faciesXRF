# WASA-faciesXRF
It is a research project in the framework of Wadden Sea Archive Project (WASA). We collect the XRF profiles and core descriptions of 92 sediment cores from tidal flats, channels, and off-shore around the island of Norderney in order to develop an automatic facies recognition model. It is also an approach to utilize machin learning and data analysis methods (SciPy ecosystem in Python) on the geosciences data. The works are conducted by jupyter notebook and python script. The jupyter notebooks are mainly used for analyzing data and some trials, while the python scripts are the heacy works submitted to the high performance computation in University of Bremen. 

## Why open this project on Github
I'm a PhD student started this development from spring 2019. Our research group is GEOPOLAR, University of Bremen. During this period, some results are achieved. I think it's time to share my workflow and scripts with the open source community because I've been benifited a lot from the community. 

## General workflow
Itrax-XRF core scanner produces raw spectral files (I would just call them spectra) and elemental intensities file (result.txt) quickly, tuned using default setting file in each run. Among the steps, there are many bugs caused by inconsistent naming and manual processing during scanning. I only upload the main scripts and summarize the steps here, otherwise it will be too messy. For the detailed steps, please check commit log.<br> 
1. I adopt the core-section information from WASA_Umrechnung_Kerntiefen_20190313.xlsx and read all the result.txt files to construct a table having elemental intensities and information for each run (e.g., directroy of spectral file, MSE, and validity) by applying `rawdata_preparation.py`. The composite depth is calculated and asigned.
2. I adopt the table produced by `rawdata_preparation.py` to clean out bad quality scanning points by applying`rawdata_cleaning.py`. Later on, the composite_id is built by combining core and composite depth to be index of each scanning point
3. I use `spectradata_preparation.py` to adopt the new directories of spectral files recorded in the table produced by `rawdata_cleaning.py` and construct a cleaned table of spectra.
4. The spectra table is used for clustering analysis before using Q-spec for reprocessing. It is becauss that it may not be suitable to use same tuning setting for reprocessing if the sediments have significant different. `spectradata_clustering.py` tries threee clustering algorithms: Kmeans, HDBSCAN, and agglomerative clustering (Ward). In the end, the table is polished again because the clustering result reveals the existence of data scanned by Mo tube instead of the selected Cr tube. 
5. The data scanned by Cr tube don't have significant from the 3 PCs space, so they are randomly shuffled into 7 clusters. The spectral files are copied to separated folders and sumspectrum.spe for each folder is created by `prepare_fore_reprocess.py`. Subsequently, the spectral files are reprocessed by Q-spec software in each folder. New result.txt files (i.e., elemntal intensities) are produced. One of reasons to separate spectral files into folder for reprocessing is that the memory limit of Q-spec. Actually, the spectral files are also reprocessed without shuffling. But the result has a bit higher overall MSE comparing to that of the shuffled step, so I don't include the script. 
6. The third time of polishing data based on the reprocessed elemental intensities. `reprocess_workflow.ipynb` marks down the steps for analyzing those reprocessed data and the cleaning. After this, the cleaned tables for spectra and reprocessed elemental intensities are established. The infos are extracted to another table. Data length: 170470
7. Update the database (three tables: elemental intensities, spetra, and infos) due to the update of composite depth, section depth and further infos. `update_database.ipynb`. Data length: 170436
8. Twelve elements (Si, S, Cl, K, Ca, Ti, Fe, Br, Rb, Sr, Zr, Ba) are selected by `select_element.ipynb`.
9. The labels of facies are digitalized and built as an array. There are several versions of the labels along the time I tried ML. They are simplified and arranged together in `build_labels.ipynb`.
10. The elemental intensities of XRF is focused first. The rolling trick to capture the composite characteristic of facies is applied in `ML_element_01.ipynb`.
11. Logistic regression classifier (lr) is applied to the rolled elemental data. The attepmts with and without PCA as a transformation is tested. The gridsearch results are visualized. The workflow and some problems are addressed in `ML_element_02.ipynb`. `grid_lr.py`, `split.py`, and `submit_lr.sh` are used for this step.
1. RBF SVC classifier (svc) is applied to the rolled elemental data. It's a Gaussian kernel support vector machine. The attepmts with and without PCA as a transformation is tested. The gridsearch results are visualized. The workflow and some problems are addressed in `ML_element_03.ipynb`. `grid_svc.py`, `split.py`, and `submit_svc.sh` are used for this step.
1. Random Forest (rf) classifier is applied to the rolled elemental data. The attepmts with and without PCA as a transformation is tested. The gridsearch results are visualized. The workflow and some problems are addressed in `ML_element_03.ipynb`. `grid_rf.py`, `split.py`, and `submit_rf.sh` are used for this step.
1. Visualize and evaluate the performance of the three optimized models (lr, svc, rf) using test set (`ML_element_04.ipynb`). It turns out the dataset needs to be modified and the workflow has to be carried out again.
1. Redo the workflow (11, 12, 13) using the updated elemtenal data and evaluate performance of models (`ML_element_05.ipynb`).
1. Further optimization and model evaluation for the rf approach (`ML_element_06.ipynb`). 
2. Redesign the data splitting strategy. The dataset is splitted into training, dev (development) and test sets in later on steps by importing functions in `split.py`. The test set is kept to the real end and for future applications to compare. I'm using only dev set to optimize our model. 
3. Apply error analysis on the dev set. I print out two core sections having most errors (wrong predictions) in each facies and manually determine the eroor categories. This analysis is iteratively processed to get logical understanding. The idea of error analysis is well described in "[Machine Learning Yearning](https://d2wvfoqc9gyqzf.cloudfront.net/content/uploads/2018/09/Ng-MLY01-13.pdf)", written by Andrew Ng. The works are carried out in `ML_element_08.ipynb` and `ML_element_09.ipynb`. The results are useful for discussing what reasons obstacle our models' preformance and for further optimizing.
4. I compare the models' performance on three kinds of data to see the benifit of feature engineering. They are the raw data, rolling data and image-like data. The image-like data has the same logic as the rolling data but instead of representing each data point by the mean and s.d., I include all adjacent data as a 2D data block to represent each data point. 
    - The models are retrained by:<br> 
    `submit_raw.sh` `grid_raw_lr.py` `grid_raw_svc.py` `grid_raw_rf.py`<br>
    `submit.sh` `grid_lr.py` `grid_svc.py` `grid_rf.py`<br>
    `submit_2d.sh` `grid_2d_lr.py` `grid_2d_svc.py` `grid_2d_rf.py`  
    P.S. `create_2d_data.py` and `ML_element_11.ipynb` is used to create the image-like data<br>
    - The models' performance is evaluated and visualized in:<br>
    `ML_element_10.ipynb`<br>
    `produce_2d_evaluations.py` `ML_element_12.ipynb`<br>
    `produce_roll_evaluations.py` `ML_element_13.ipynb`<br>
5. The confidence score of the SVC model on the predictions are illustrated in `ML_element_07.ipynb`

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. If you're a newby student want to work on large geochemical and sedimentary dataset, you might find something interesting in this project. If you're a experienced data scientist, you might find immature way of analyzing in this project. After all, I'm willing to share my experience to you.

Since it's my first time to open source, there are still many details need to be improved.
```python
print('Hello world!')
```
