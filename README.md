# WASA-faciesXRF
It is a research project in the framework of Wadden Sea Archive Project (WASA). We collect the XRF profiles and core descriptions of 92 sediment cores from tidal flats, channels, and off-shore around the island of Norderney in order to develop an automatic facies recognition model. It is also an approach to utilize machine learning and data analysis methods (SciPy ecosystem in Python) on the geosciences data. The works are conducted by jupyter notebook and python script. The jupyter notebooks are mainly used for analyzing data and some trials, while the python scripts are the heavy works submitted to the high performance computation in University of Bremen. 

## Why open this project on Github
I'm a PhD student started this development from spring 2019. There are GEOPOLAR (University of Bremen) and ENL (National Taiwan University) support my study jointly. During this period, some results are achieved. I think it's time to share my workflow and scripts with the open source community because I've been benifited a lot from the community. 

## General workflow
### Data preparation
Itrax-XRF core scanner produces raw spectral files (I would just call them spectra) and elemental intensities file (result.txt) quickly, tuned using default setting file in each run. Among the steps, there are many bugs caused by inconsistent naming and manual processing during scanning. I only upload the main scripts and summarize the steps here, otherwise it will be too messy. For the detailed steps, please check commit log.<br> 
1. I adopt the core-section information from WASA_Umrechnung_Kerntiefen_20190313.xlsx and read all the result.txt files to construct a table having elemental intensities and information for each run (e.g., directroy of spectral file, MSE, and validity) by applying `rawdata_preparation.py`. The composite depth is calculated and asigned.
2. I adopt the table produced by `rawdata_preparation.py` to clean out bad quality scanning points by applying`rawdata_cleaning.py`. Later on, the composite_id is built by combining core and composite depth to be index of each scanning point.
3. I use `spectradata_preparation.py` to adopt the new directories of spectral files recorded in the table produced by `rawdata_cleaning.py` and construct a cleaned table of spectra.
4. The spectra table is used for clustering analysis before using Q-spec for reprocessing. It is becauss that it may not be suitable to use same tuning setting for reprocessing if the sediments have significant different. `spectradata_clustering.py` tries three clustering algorithms: Kmeans, HDBSCAN, and agglomerative clustering (Ward). In the end, the table is polished again because the clustering result reveals the existence of data scanned by Mo tube instead of the selected Cr tube. 
5. The data scanned by Cr tube don't have significant difference from the 3 PCs space, so they are randomly shuffled into 7 clusters. The spectral files are copied to separated folders and sumspectrum.spe for each folder is created by `prepare_for_reprocess.py`. Subsequently, the spectral files are reprocessed by Q-spec software in each folder. New result.txt files (i.e., elemntal intensities) are produced. One of reasons to separate spectral files into folder for reprocessing is that the memory limit of Q-spec. Actually, the spectral files are also reprocessed without shuffling. But the result has a bit higher overall MSE comparing to that of the shuffled step, so I don't include the script. 
6. The third time of polishing data based on the reprocessed elemental intensities. `reprocess_workflow.ipynb` marks down the steps for analyzing those reprocessed data and the cleaning. After this, the cleaned tables for spectra and reprocessed elemental intensities are established. The infos are extracted to another table. Data length: 170470
7. Update the database (three tables: elemental intensities, spetra, and infos) due to the update of composite depth, section depth and further infos. `update_database.ipynb`. Data length: 170436
8. Twelve elements (Si, S, Cl, K, Ca, Ti, Fe, Br, Rb, Sr, Zr, Ba) are selected by `select_element.ipynb`.
9. The labels of facies are digitalized and built as an array. There are several versions of the labels along the time I tried ML. They are simplified and arranged together in `build_labels.ipynb`.
```
```
### ML implementation
1. The elemental intensities of XRF is focused first. The rolling trick to capture the composite characteristic of facies is applied in `ML_element_01.ipynb`.
1. Logistic regression classifier (lr) is applied to the rolled elemental data. The attepmts with and without PCA as a transformation is tested. The gridsearch results are visualized. The workflow and some problems are addressed in `ML_element_02.ipynb`. `grid_lr.py`, `split.py`, and `submit_lr.sh` are used for this step.
1. RBF SVC classifier (svc) is applied to the rolled elemental data. It's a Gaussian kernel support vector machine. The attepmts with and without PCA as a transformation is tested. The gridsearch results are visualized. The workflow and some problems are addressed in `ML_element_03.ipynb`. `grid_svc.py`, `split.py`, and `submit_svc.sh` are used for this step.
1. Random Forest (rf) classifier is applied to the rolled elemental data. The attepmts with and without PCA as a transformation is tested. The gridsearch results are visualized. The workflow and some problems are addressed in `ML_element_03.ipynb`. `grid_rf.py`, `split.py`, and `submit_rf.sh` are used for this step.
1. Visualize and evaluate the performance of the three optimized models (lr, svc, rf) using test set (`ML_element_04.ipynb`). It turns out the dataset needs to be modified and the workflow has to be carried out again.
1. Redo the workflow (11, 12, 13) using the updated elemtenal data and evaluate performance of models (`ML_element_05.ipynb`).
1. Further optimization and model evaluation for the rf approach (`ML_element_06.ipynb`). 
1. Redesign the data splitting strategy. The dataset is splitted into training, dev (development) and test sets for later on steps by importing functions in `split.py`. The test set is kept to the real end and for future applications to compare. I'm using only dev set to optimize our model. 
1. Apply error analysis on the dev set. I print out two core sections having most errors (wrong predictions) in each facies and manually determine the error categories. This analysis is iteratively processed to get logical understanding. The idea of error analysis is well described in "[Machine Learning Yearning](https://github.com/ajaymache/machine-learning-yearning.git)", written by Andrew Ng. The works are carried out in `ML_element_08.ipynb` and `ML_element_09.ipynb`. The results are useful for discussing what reasons obstacle our models' preformance and for further optimizing.
1. I compare the models' performance on three kinds of data to see the benefit of feature engineering. They are the raw data, rolling data and image-like data. The image-like data has the same logic as the rolling data but instead of representing each data point by the mean and s.d., I include all adjacent data as a 2D data block to represent each data point. 
    - The models are retrained by:<br> 
    `submit_raw.sh` `grid_raw_lr.py` `grid_raw_svc.py` `grid_raw_rf.py`<br>
    `submit.sh` `grid_lr.py` `grid_svc.py` `grid_rf.py`<br>
    `submit_2d.sh` `grid_2d_lr.py` `grid_2d_svc.py` `grid_2d_rf.py`  
    P.S. `create_2d_data.py` and `ML_element_11.ipynb` is used to create the image-like data<br>
    - The models' performance is evaluated and visualized in:<br>
    `ML_element_10.ipynb`<br>
    `produce_2d_evaluations.py` `ML_element_12.ipynb`<br>
    `produce_roll_evaluations.py` `ML_element_13.ipynb`<br>
1. The confidence score of the SVC model on the predictions are illustrated in `ML_element_07.ipynb`
1. `evaluation.py` is developed to include some evaluation functions.
1. `post_process.ipynb` and `post_process.py` are developed to implement post smoothing on the models' prediction, which we expect to reduce fragmentation and increase accuracy. There are two ways of smoothing: simple and sophisticate. After a long time developing, the simple smoothing actually does better job than the sophisticating smoothing. But, both ways can't increase the accuracy noticibly, which means the sediments are mostly misclassified in big chunks.
1. The integrated model (SVC+post smoothing) is applied to the test set to see the final performance: `SVC_smooth_test.py` and `ML_element_14.ipynb`.
1. Dig more deeper to the machine confidence level (`machine_confidence.ipynb` and `build_prob_svc.py`). In the meantime, I find a major mistake in the previous model building so the model is rebuilt (`build_final_model.py`).
```
```
### ML implementation: reclassifying labels
It's a major changing step. I finally decide to face the biggest error in our model, "the description (y) isn't really good at the begining." This error contributes most in both the error analysis and my mind. Without Dirk's help and push, I wouldn't do it because it requires doing all over the ML workflow again using the reclassified facies label. Even though I have all the codes and experience, it still needs large effort and time to redo. In summary, 1/8 of the sediments are reclassified with a more careful manner and reasonable simplification. The data quality is more exquisite but still in a large quatity. I then use these new labels and subset of data to build our models. 
1. Develope scripts to adopt the new label and redo the workflow (`ML_element_15.ipynb` and `wasafacies.py`).
    - The models are trained by:<br> 
    `submit_raw.sh` `grid_r_raw_lr.py` `grid_r_raw_svc.py` `grid_r_raw_rf.py`<br>
    `submit.sh` `grid_r_lr.py` `grid_r_svc.py` `grid_r_rf.py`<br>
    `submit_2d.sh` `grid_2d.py` (including three algorithms already)  `grid_2d_rf.py` (for customaization) 
1. Visualize and investigate the performance of models. Some data analyses are carried out and new sections are supplemented. `ML_element_16.ipynb` and the later part of `ML_element_15.ipynb`.
1. An serious error is found: "Pleistocene marine" and "Pleistocene moraine” are sharing the same abbreviation, which I used as y, so I modify the abbreviation and use numbers as y to represent the facies in begining. This overcomes the inconsistent facterization issue in different data representations previously had. Also, the sections of sandflat are adjusted. The modifications and redo results are recorded in `ML_element_17.ipynb`.
    - The models are trained by:<br> 
    `submit_raw.sh` `grid_r_raw_lr.py` `grid_r_raw_svc.py` `grid_r_raw_rf.py`<br>
    `submit.sh` `grid_r_lr.py` `grid_r_svc.py` `grid_r_rf.py`<br>
    `submit_2d.sh` `grid_2d.py` (including three algorithms already)  
1. Model's confidence level is explorated in `ML_element_18.ipynb`.

## Paper preparation
- `prepare_paper_01.ipynb`: polishing figures for the results of the first ML attempt.
- `prepare_paper_02.ipynb`: polishing figures for the results of the second ML attempt (reclassifying labels) 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. If you're a newby student want to work on large geochemical and sedimentary dataset, you might find something interesting in this project. If you're a experienced data scientist, you might find immature way of analyzing in this project. After all, I'm willing to share my experience to you.

Since it's my first time to open source, there are still many details need to be improved.
```python
print('Hello world!')
```
