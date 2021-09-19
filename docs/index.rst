Defect Detection and Quantification Toolbox (DDQT)
==================================================

The domain of `Nondestructive Testing (NDT)` and `Structural Health
Monitoring (SHM)` comprise of techniques that are used to evaluate
the state of health of a structure without causing any damage to the
structure being inspected. Typical examples of structures being inspected
include aircraft components, bridges, nuclear reactors, etc. `Defect
Detection and Quantification Toolbox (DDQT)` aims to provide a framework to
automate the routine tasks for researchers in the field of `NDT` and `SHM`.

In the fields of `NDT` and `SHM`, experiments are performed using a variety
of modalities like Ultrasound, Infrared Thermography, X-Rays, etc.
`Matlab` is often used to perform experiments. Typically, the resulting data
is very often a 3D dataset comprising of time axis and 2D spatial axis. Once
the data is obtained, a variety of visualization checks are performed.
Following this, features are created at each and every spatial location in
the time series. Some examples of the types of features are based on time
series, gradients, spatial filters, etc. At this stage, to avoid the `curse
of dimensionality`, a feature reduction step is performed using
`PCA` in case of `unsupervised defect detection`. Once the subset of
meaningful features is identified, the defect regions are detected using a
statistical distance metric like `Mahalanobis distance`. In `DDQT`, we also
propose a realatively newer method to identify defects using `Isolation
Forests`. Following this, the performance of the feature space and detection
algorithms is quantified using `Receiver Operating Curves (ROC)` curves
compared to the raw data. The performance is quantified using commonly used
classification metrics such as `True Positive Rate (TPR)`, `False Positive
Rate (FPR)` and `Area Under Curve (AUC)`. A visual representation of these
metrics is presented using charts to aid the user.

While there are clearly defined steps that researchers in field of `NDT` and
`SHM` often pursue, to the best of our knowledge there are no offerings that
provide a framework so researchers can mainly focus on the feature space and
tweaking the defect detection algorithms. With very minimal edits to the
driver program, researchers can visualize the results with ease and focus on
the underlying `physics` to improve the performance of defect detection
instead of spending much time and effort on setting up the pipeline process.
 

**In Summary, `DDQT` can be used for;**

  * Reading in Matlab data
  * Visualizing data
  * Creating features in the time and spatial domain
  * Feature reduction using PCA
  * Identifying defects using Mahalanobis distance and Isolation Forest
  * Quantifying results using ROC curves
  * Visualizing outcomes

There are numerous avenues to enhance this toolbox. I welcome any
contributions to this program. **Some possible areas that could use
improvements are;**

  * Improvements in feature space
  * Improvements to defect detection algorithms
  * Coding enhancements
  * Documentation enhancements
  * Currently, only certain time stamps are used in calculating
    computationally intensive features. There is scope to write more
    computationally efficient code to handle more time stamps (if not
    everything...)
  * Possibility of including circular defects - currently, defects are
    defined using polygon vertices

If you would like to collaborate with me in improving this toolbox or if you
would like to provide sample data, please reach out to me at

::

   >>>my_first_name = 'arun'
   >>>print(str(my_first_name) + 'mano121@outlook.com')

Feel free to fork and add any enhancements, and let me know if a pull request
is needed to merge the changes. 

If you use this work in your research, please cite using;

::

    @software{ArunManohar_20210322,
      author       = {Arun Manohar},
      title        = {{Defect Detection and Quantification Toolbox (DDQT)}},
      month        = mar,
      year         = 2021,
      publisher    = {Zenodo},
      version      = {v0.1.0},
      doi          = {10.5281/zenodo.4627984},
      url          = {https://doi.org/10.5281/zenodo.4627984}
    } 

Thank you!


.. toctree::
   :maxdepth: 2
   :hidden:

   readme
   getting_started
   citation
   modules
   license
   acknowledgements
   contact
