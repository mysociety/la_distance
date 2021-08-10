# Local authority distance comparison

This repository contains Jupyter notebooks exploring the 'distance' between local authority on either generic or emissions specific features of local authorities. 

The goal is to explore and develop methods for drawing connections between 'similar' authorities to improve learning between authorities with similar problems.

This is produced as part of [mySociety's Climate progrmame](https://www.mysociety.org/climate/). 

Key notebooks:

* [emissions.v1.ipynb](emissions.v1.ipynb) - exploring how the BEIS emissions dataset can be used to compare local authorities. 
* [deprivation.ipynb](deprivation.ipynb) - exploring how deprivation/density groups authorities, and how this differs from emissions or physical distance. 
* [lookup_demo.ipynb](lookup_demo.ipynb) - Demo of an emissions distance comparison tool (described in the emissions notebook).

Outputs from these processes are stored in `data\outputs`. For each comparison, this contains a file showing the distance (unit is meaningless, but can be used to rank), and a file showing the results of a clustering process. 

## Licencing

Shapefiles for calcualting area and physical distance derived from [Boundary-Line](https://www.ordnancesurvey.co.uk/business-government/products/boundaryline) and [OSNI Local Government districts](https://www.opendatani.gov.uk/dataset/osni-open-data-largescale-boundaries-local-government-districts-2012).

Source datasets are Open Government licence unless otherwise stated.

Output datasets are licensed under a Creative Commons Attribution 4.0 International License.

Code and scripts are licenced under a MIT Licence.

Raincloud plots citation: Allen M, Poggiali D, Whitaker K et al. Raincloud plots: a multi-platform tool for robust data visualization [version 2; peer review: 2 approved]. Wellcome Open Res 2021, 4:63. DOI: [10.12688/wellcomeopenres.15191.2](https://wellcomeopenresearch.org/articles/4-63/v2)

