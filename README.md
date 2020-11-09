# WASA-faciesXRF
It is a research project in the framework of Wadden Sea Archive Project (WASA). We collect the XRF profiles and core descriptions of 92 sediment cores from tidal flats, channels, and off-shore around the island of Norderney in order to develop an automatic facies recognition model. It is also an approach to utilize machin learning and data analysis methods (SciPy ecosystem in Python) on the geosciences data. The works are conducted by jupyter notebook and python script. The jupyter notebooks are mainly used for analyzing data and some trials, while the python scripts are the heacy works submitted to the high performance computation in University of Bremen. 

## Why open this project on Github
I'm a PhD student started this development from spring 2019. Our research group is GEOPOLAR, University of Bremen. During this period, some results are achieved. I think it's time to share my workflow and scripts with the open source community because I've been benifited a lot from the community. 

## General workflow
Itrax-XRF core scanner produces raw spectral files (I would just call them spectra) and elemental intensities file (result.txt) quickly tuned using default setting file in each run. Among the steps, there are many bugs caused by inconsistent naming and manual processing during scanning. I only updload the main scripts, otherwise it will be too messy.<br> 
1. I adopt the core-section information from WASA_Umrechnung_Kerntiefen_20190313.xlsx and read all the result.txt files to construct a table having elemental intensities and information for each run (e.g., directroy of spectral file, MSE, and validity) by applying `rawdata_preparation.py`. The composite depth is calculated and asigned.
2. I adopt the table produced by `rawdata_preparation.py` to clean out bad quality scanning points by applying`rawdata_cleaning.py`. Later on, the composite_id is built by combining core and composite depth to be index of each scanning point
3. I use `spectradata_preparation.py` to adopt the new directories of spectral files recorded in the table produced by `rawdata_cleaning.py` and construct a cleaned table of spectra.
4. The spectra table is used for clustering analysis before using Q-spec for reprocessing. It is becauss that it may not be suitable to use same tuning setting for reprocessing if the sediments have significant different. `spectradata_clustering.py` tries threee clustering algorithms: Kmeans, HDBSCAN, and agglomerative clustering (Ward). In the end, the table is polished again because the clustering result reveals the existence of data scanned by Mo tube instead of the selected Cr tube. 
5. The data scanned by Cr tube don't have significant from the 3 PCs space, so they are randomly shuffled into 7 clusters.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. If you're a newby student want to work on large geochemical and sedimentary dataset, you might find something interesting in this project. If you're a experienced data scientist, you might find immature way of analyzing in this project. After all, I'm willing to share my experience to you.

Since it's my first time to open source, there are still many details need to be improved.
```python
print('Hello world!')
```
