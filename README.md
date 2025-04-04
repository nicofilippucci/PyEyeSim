# PyEyeSim

this is a Python library for eye-movement analysis, visualization and comparison, with a focus on scanpath comparison.

The ultimate goal of the library is to make advanced fixation map statistics (eg: entropy) and scanpath comparison  accesible (hidden markov model based, and saccade direction based).

The library also provides general descripitve statistics about eye-movements. It is intended to work with ordered fixation data. (a row for each fixation), that is loaded into a pandas dataframe.   

Additionaly, easy visualizations about the statistics (overall stats, stimulus based stats, within trial progression) and heatmaps are also provided. 

three main scanpath similarity functionalities:

1. Within group similarity  (for a single group of observers in a single condition)
2. Between condition similarity (for single group of observers, observing the same stimuli in two conditions)
3. Between group similarity (for two groups of observers observing the same stimuli)


The library started to develop for use in art perception studies, therefore, there is an emphasis on stimulus based eye-movement comparison.


#### Installation:
in the terminal/anaconda prompt

`pip install pyeyesim`

OR 

if you cloned the library and are in the root folder of the library using the terminal (mac) or anaconda prompt (windows), you can install it, using the command: 
**" pip install -e . "**


#### Demo:
for examples of using the library, see the PyEyeSimDemo.ipynb in the Notebooks folder


#### Dependencies:
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

##### for full funcionality
  scikit-image
  hmmlearn
  xarray

##### on pip
https://pypi.org/project/pyeyesim/



#### for the demo notebooks, the dataset is from:
Wilming, N., Onat, S., Ossandón, J. P., Açık, A., Kietzmann, T. C., Kaspar, K., ... & König, P. (2017). An extensive dataset of eye movements during viewing of complex images. Scientific data, 4(1), 1-11.

to run the demo download the CSV from:
https://ucloud.univie.ac.at/index.php/s/3FF3iqZDzc3MxK9

images can be downloaded from:
https://datadryad.org/stash/dataset/doi:10.5061/dryad.9pf75
only download images:  Stimuli_8.zip

