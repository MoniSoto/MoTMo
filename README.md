# MoTMo
Some analysis from the outcome of the MoTMo agent-based model

- Link for downloading the dataset: https://www.zib.de/tes-data-sets/data/motmo.zip
- Original paper of the ABM: https://globalclimateforum.org/wp-content/uploads/2021/04/GCFwp2_2021.pdf
- The file `MoTMo-scenarioDataExplainer.pdf` also has a summary description of the dataset (it is included in the `motmo.zip` file if you download it).

Inside the `jupyter_notebooks` folder you can find a notebook called `1data_challenge` that includes the results of the sensitivity analysis (not analyzed) for each of the categories. We have also created a module called `MoTMo.py` which was imported into the notebook as `mo`. You can check out the functions defined there, some of them have an explanation or are self-explanatory.

More recently, we have added another notebook `2data_chall_indiv_options` that includes a more detailed analysis of the individual options.

We have also included another notebook called `SALib_analysis` which is the result for total emissions of a Python package called `SALib` that makes these analysis, but we made a further transformation of the input space. It might also be interesting to take a look at it, but it is not very tidy and we did not included it into the report.

For any questions, do not hesitate to contact us.
